#  pip install ortools
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

import cp_model

# -----------------------------
# Domain model
# -----------------------------
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

@dataclass(frozen=True)
class School:
    id: str
    name: str

@dataclass(frozen=True)
class Student:
    id: str
    name: str
    school_id: str
    impairment: str
    duration_slots: int = 1     # number of consecutive timeslots per session (default 1 block)
    must_meet_this_week: bool = True  # False -> biweekly/optional this week

@dataclass(frozen=True)
class AvailabilityBlock:
    """An availability window for a STUDENT (teacher-provided) OR for SLP-blocked times.
    Use day_idx in [0..4], start_slot inclusive, end_slot exclusive.
    """
    day_idx: int
    start_slot: int
    end_slot: int

@dataclass
class ProblemData:
    timeslot_minutes: int                 # e.g., 30
    day_start_minutes: int                # e.g., 8*60 for 8:00
    day_end_minutes: int                  # e.g., 15*60 + 30 for 3:30pm
    schools: List[School]
    students: List[Student]
    # Student availability: student_id -> list of student-available windows
    student_availability: Dict[str, List[AvailabilityBlock]]
    # Global SLP blocked windows (cannot schedule anyone)
    slp_blocked: List[AvailabilityBlock]

# -----------------------------
# Helper: discrete time grid
# -----------------------------
def build_time_grid(day_start_minutes: int, day_end_minutes: int, slot_minutes: int) -> int:
    total = (day_end_minutes - day_start_minutes) // slot_minutes
    if (day_end_minutes - day_start_minutes) % slot_minutes != 0:
        raise ValueError("Day length must be divisible by slot_minutes.")
    return total

def slot_label(day_idx: int, slot_idx: int, pd: ProblemData) -> str:
    start_min = pd.day_start_minutes + slot_idx * pd.timeslot_minutes
    h = start_min // 60
    m = start_min % 60
    return f"{DAYS[day_idx]} {h:02d}:{m:02d}"

# -----------------------------
# Feasibility masks
# -----------------------------
def build_student_feasible_mask(pd: ProblemData) -> Dict[str, List[List[bool]]]:
    """Return mask[student_id][day][slot] -> True if student *could* start at slot
    (respecting student availability, SLP blocked, and session duration)."""
    T = build_time_grid(pd.day_start_minutes, pd.day_end_minutes, pd.timeslot_minutes)
    # Start with all True, then cut by constraints
    mask = {s.id: [[True]*T for _ in range(5)] for s in pd.students}

    # Apply SLP global blocked times
    blocked = [[False]*T for _ in range(5)]
    for b in pd.slp_blocked:
        for t in range(b.start_slot, b.end_slot):
            blocked[b.day_idx][t] = True

    # Per student: intersect availability and ensure full duration fits inside any window
    for s in pd.students:
        # Start by zeroing everything, then enable in allowed windows
        allowed = [[False]*T for _ in range(5)]
        for win in pd.student_availability.get(s.id, []):
            for t in range(win.start_slot, min(win.end_slot, T)):
                allowed[win.day_idx][t] = True

        # Now build start feasibility (duration_contiguous, and not hitting SLP blocks)
        for d in range(5):
            for t in range(T):
                ok = True
                # duration must fit
                if t + s.duration_slots > T:
                    ok = False
                else:
                    # All constituent slots must be allowed and not SLP-blocked
                    for k in range(s.duration_slots):
                        if not allowed[d][t+k] or blocked[d][t+k]:
                            ok = False
                            break
                mask[s.id][d][t] = ok
    return mask

# -----------------------------
# Solver
# -----------------------------
class SLPScheduler:
    def __init__(self, pd: ProblemData):
        self.pd = pd
        self.T = build_time_grid(pd.day_start_minutes, pd.day_end_minutes, pd.timeslot_minutes)
        self.school_index = {s.id: i for i, s in enumerate(pd.schools)}
        self.students_by_school: Dict[str, List[Student]] = defaultdict(list)
        self.students_by_impairment: Dict[str, List[Student]] = defaultdict(list)
        for s in pd.students:
            self.students_by_school[s.school_id].append(s)
            self.students_by_impairment[s.impairment].append(s)
        self.mask = build_student_feasible_mask(pd)

    def solve(self, time_limit_seconds: Optional[int] = 30):
        model = cp_model.CpModel()

        # x[s,d,t] = 1 if student s starts a session at day d, slot t
        x: Dict[Tuple[str,int,int], cp_model.IntVar] = {}

        # For grouping constraints:
        # z[d,t,school] = 1 if any session runs at (d,t) at that school (any impairment)
        z: Dict[Tuple[int,int,str], cp_model.IntVar] = {}

        # y[d,t,school,imp] = 1 if impairment 'imp' is active at (d,t,school)
        # Used to enforce homogeneity and size cap by impairment.
        y: Dict[Tuple[int,int,str,str], cp_model.IntVar] = {}

        # Convenience sets
        impairments: Set[str] = set(s.impairment for s in self.pd.students)
        school_ids = [s.id for s in self.pd.schools]

        # Create variables
        for s in self.pd.students:
            for d in range(5):
                for t in range(self.T):
                    if self.mask[s.id][d][t]:
                        x[(s.id, d, t)] = model.NewBoolVar(f"x_{s.id}_{d}_{t}")
        for d in range(5):
            for t in range(self.T):
                for sc in school_ids:
                    z[(d,t,sc)] = model.NewBoolVar(f"z_{d}_{t}_{sc}")
                    for imp in impairments:
                        y[(d,t,sc,imp)] = model.NewBoolVar(f"y_{d}_{t}_{sc}_{imp}")

        # 1) Each student scheduled at most once per week (exactly once if must_meet_this_week)
        for s in self.pd.students:
            vars_for_s = [x[(s.id,d,t)] for d in range(5) for t in range(self.T) if (s.id,d,t) in x]
            if vars_for_s:
                if s.must_meet_this_week:
                    model.Add(sum(vars_for_s) == 1)
                else:
                    model.Add(sum(vars_for_s) <= 1)
            else:
                # If no feasible window exists but must meet, solver will declare infeasible.
                if s.must_meet_this_week:
                    # Create an impossible constraint to surface infeasibility fast:
                    model.Add(0 == 1)

        # 2) Link z and y to x (occupancy & impairment activation)
        # Also enforce group size ≤ 3 *and* homogeneity per (day,slot,school).
        # We treat a session covering multiple consecutive slots by marking every covered slot occupied.
        for d in range(5):
            for t in range(self.T):
                for sc in school_ids:
                    # If any student session covers (d,t) at school sc, z[d,t,sc] = 1
                    covering_students = []
                    for s in self.students_by_school[sc]:
                        for start in range(max(0, t - (s.duration_slots - 1)), t + 1):
                            if (s.id, d, start) in x and start + s.duration_slots > t:
                                covering_students.append(x[(s.id, d, start)])
                    if covering_students:
                        print(covering_students)
                        model.Add(z[(d,t,sc)] >= sum(covering_students) * 0.0001)  # z >= any(covering)
                        # z <= sum(...) but cap to 1 via boolean nature:
                        model.Add(sum(covering_students) <= len(covering_students) * z[(d,t,sc)])
                    else:
                        # No one can cover this slot; force z=0
                        model.Add(z[(d,t,sc)] == 0)

                    # Impairment activation variables y[d,t,sc,imp]
                    for imp in impairments:
                        covering_by_imp = []
                        for s in self.students_by_school[sc]:
                            if s.impairment != imp:
                                continue
                            for start in range(max(0, t - (s.duration_slots - 1)), t + 1):
                                if (s.id, d, start) in x and start + s.duration_slots > t:
                                    covering_by_imp.append(x[(s.id, d, start)])
                        if covering_by_imp:
                            model.Add(y[(d,t,sc,imp)] >= sum(covering_by_imp) * 0.0001)
                            model.Add(sum(covering_by_imp) <= len(covering_by_imp) * y[(d,t,sc,imp)])
                        else:
                            model.Add(y[(d,t,sc,imp)] == 0)

                    # Homogeneity: at most ONE impairment active in a (d,t,sc)
                    model.Add(sum(y[(d,t,sc,imp)] for imp in impairments) <= 1)

                    # Group size ≤ 3 per impairment (and since homogeneity, per slot overall)
                    for imp in impairments:
                        # Count students present at that (d,t,sc,imp)
                        count_here = []
                        for s in self.students_by_school[sc]:
                            if s.impairment != imp:
                                continue
                            for start in range(max(0, t - (s.duration_slots - 1)), t + 1):
                                if (s.id, d, start) in x and start + s.duration_slots > t:
                                    count_here.append(x[(s.id, d, start)])
                        if count_here:
                            model.Add(sum(count_here) <= 3)

        # 3) "Do not drive to a school more than once per day":
        #     Enforce that for each (day, school), the occupied slots form at most one contiguous block.
        #     We count 0->1 transitions in z[d, t, school] and force that count ≤ 1.
        for d in range(5):
            for sc in school_ids:
                transitions = []
                # t = 0 transition from implicit 0
                u0 = model.NewBoolVar(f"u_{d}_0_{sc}")
                model.Add(u0 >= z[(d,0,sc)])
                model.Add(u0 <= z[(d,0,sc)])
                transitions.append(u0)
                # For t >= 1: u[t] = 1 if z[t-1]=0 and z[t]=1
                for t in range(1, self.T):
                    u = model.NewBoolVar(f"u_{d}_{t}_{sc}")
                    # u <= z[t]
                    model.Add(u <= z[(d,t,sc)])
                    # u <= 1 - z[t-1]
                    model.Add(u <= 1 - z[(d,t-1,sc)])
                    # u >= z[t] - z[t-1]
                    # (linearized by two constraints)
                    model.Add(u >= z[(d,t,sc)] - z[(d,t-1,sc)])
                    model.Add(u >= 0)
                    transitions.append(u)
                # At most one 0->1 transition -> at most one contiguous visit block
                model.Add(sum(transitions) <= 1)

        # 4) (Optional but useful) Soft objective:
        #    - Prefer batching students of the same school & impairment into as few slots as possible
        #    - Prefer scheduling optional (biweekly) students only if there is room
        # Implement a simple objective:
        #   maximize (sum scheduled must_meet) + 0.1 * sum scheduled optional - 0.01 * total distinct occupied slots
        scheduled_must = []
        scheduled_opt = []
        distinct_slots = []
        for s in self.pd.students:
            vars_for_s = [x[(s.id,d,t)] for d in range(5) for t in range(self.T) if (s.id,d,t) in x]
            if vars_for_s:
                if s.must_meet_this_week:
                    scheduled_must.extend(vars_for_s)
                else:
                    scheduled_opt.extend(vars_for_s)
        for d in range(5):
            for t in range(self.T):
                # Any school occupied at (d,t)?
                any_occ = model.NewBoolVar(f"occ_{d}_{t}")
                model.AddMaxEquality(any_occ, [z[(d,t,sc)] for sc in school_ids])
                distinct_slots.append(any_occ)

        # Objective
        model.Maximize(
            1000 * sum(scheduled_must) +               # prioritize required students
            100 * sum(scheduled_opt) -                 # then optional
            1 * sum(distinct_slots)                    # lightly penalize spread-out usage
        )

        # Solve
        solver = cp_model.CpSolver()
        if time_limit_seconds:
            solver.parameters.max_time_in_seconds = float(time_limit_seconds)
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        result = {
            "status": solver.StatusName(status),
            "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            "assignments": []  # list of dicts
        }
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for s in self.pd.students:
                for d in range(5):
                    for t in range(self.T):
                        if (s.id,d,t) in x and solver.Value(x[(s.id,d,t)]) == 1:
                            result["assignments"].append({
                                "student_id": s.id,
                                "student_name": s.name,
                                "school_id": s.school_id,
                                "impairment": s.impairment,
                                "day": DAYS[d],
                                "start_slot": t,
                                "start_label": slot_label(d, t, self.pd),
                                "duration_slots": s.duration_slots
                            })
            # Sort nicely by day, school, time
            result["assignments"].sort(key=lambda a: (DAYS.index(a["day"]), a["school_id"], a["start_slot"]))
        return result

# -----------------------------
# Example usage with sample data
# -----------------------------
def demo():
    # Define time grid: 8:00–3:30 PM in 30-min slots -> 15 slots
    pd = ProblemData(
        timeslot_minutes=30,
        day_start_minutes=8*60,
        day_end_minutes=15*60 + 30,
        schools=[
            School("A", "Lincoln Elementary"),
            School("B", "Roosevelt Elementary"),
            School("C", "Washington Elementary"),
        ],
        students=[
            Student("s1", "Ava",   "A", "Articulation", duration_slots=2, must_meet_this_week=True),
            Student("s2", "Ben",   "A", "Articulation", duration_slots=1, must_meet_this_week=True),
            Student("s3", "Cara",  "A", "Fluency",      duration_slots=1, must_meet_this_week=False),
            Student("s4", "Diego", "B", "Articulation", duration_slots=1, must_meet_this_week=True),
            Student("s5", "Ella",  "B", "Language",     duration_slots=1, must_meet_this_week=True),
            Student("s6", "Finn",  "C", "Language",     duration_slots=1, must_meet_this_week=False),
            Student("s7", "Gina",  "C", "Language",     duration_slots=1, must_meet_this_week=True),
            Student("s8", "Hank",  "C", "Language",     duration_slots=1, must_meet_this_week=True),
        ],
        student_availability={},  # fill below
        slp_blocked=[
            # SLP staff meeting: Tue 10:00–11:00 (two 30-min slots: indices 4-6)
            AvailabilityBlock(day_idx=1, start_slot=4, end_slot=6),
            # Lunch everyday 12:00–12:30 (slot index 8)
            AvailabilityBlock(day_idx=0, start_slot=8, end_slot=9),
            AvailabilityBlock(day_idx=1, start_slot=8, end_slot=9),
            AvailabilityBlock(day_idx=2, start_slot=8, end_slot=9),
            AvailabilityBlock(day_idx=3, start_slot=8, end_slot=9),
            AvailabilityBlock(day_idx=4, start_slot=8, end_slot=9),
        ],
    )

    # Helper to create a wide availability (e.g., 9:00–2:30 = slots 2..13)
    def avail(day_idx, start_hm, end_hm):
        sh, sm = start_hm
        eh, em = end_hm
        start_slot = ((sh*60 + sm) - pd.day_start_minutes) // pd.timeslot_minutes
        end_slot = ((eh*60 + em) - pd.day_start_minutes) // pd.timeslot_minutes
        return AvailabilityBlock(day_idx, start_slot, end_slot)

    # Teacher-provided availability per STUDENT
    pd.student_availability = {
        "s1": [avail(0,(9,0),(14,30)), avail(2,(9,0),(14,30))],
        "s2": [avail(0,(9,0),(14,30)), avail(3,(9,0),(14,30))],
        "s3": [avail(0,(10,0),(12,0)), avail(4,(9,0),(12,0))],  # optional this week
        "s4": [avail(1,(9,0),(13,0)), avail(3,(10,0),(14,0))],
        "s5": [avail(1,(9,0),(13,0)), avail(4,(9,0),(14,0))],
        "s6": [avail(2,(9,0),(13,0))],  # optional
        "s7": [avail(2,(9,0),(14,30)), avail(4,(9,0),(14,0))],
        "s8": [avail(2,(9,0),(14,30)), avail(4,(9,0),(14,0))],
    }

    scheduler = SLPScheduler(pd)
    result = scheduler.solve(time_limit_seconds=30)

    print("Status:", result["status"])
    print("Objective:", result["objective"])
    print("\nAssignments:")
    for a in result["assignments"]:
        print(f"  {a['start_label']}  |  {a['student_name']:5s}  | School {a['school_id']}  | {a['impairment']:12s}  | {a['duration_slots']} slot(s)")

if __name__ == "__main__":
    demo()