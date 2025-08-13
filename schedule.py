
# Sample Python software architecture using core data structures and logic

# This is a simplified prototype to illustrate how the scheduling engine might work



from typing import List, Dict, Tuple

from collections import defaultdict

import itertools



# Define core data classes

class Student:

    def __init__(self, id: int, name: str, school: str, impairment: str, availability: List[str]):

        self.id = id

        self.name = name

        self.school = school

        self.impairment = impairment

        self.availability = availability



class Group:

    def __init__(self, students: List[Student]):

        self.students = students

        self.school = students[0].school

        self.impairment = students[0].impairment

        self.common_availability = list(set.intersection(*(set(s.availability) for s in students)))



class Scheduler:

    def __init__(self, students: List[Student], blocked_times: List[str]):

        self.students = students

        self.blocked_times = set(blocked_times)

        self.schedule = defaultdict(list)  # Key: Day, Value: List of (school, time, students)



    def group_students(self) -> List[Group]:

        groups = []

        # Group students by (school, impairment)

        grouped = defaultdict(list)

        for student in self.students:

            key = (student.school, student.impairment)

            grouped[key].append(student)



        for key, students in grouped.items():

            # Generate all valid combinations of 2 or 3 students

            for r in [2, 3]:

                for combo in itertools.combinations(students, r):

                    group = Group(list(combo))

                    if group.common_availability:

                        groups.append(group)

        return groups



    def generate_schedule(self) -> Dict[str, List[Tuple[str, str, List[str]]]]:

        daily_schools = defaultdict(set)

        groups = self.group_students()

        used_times = set()



        for group in groups:

            for time in group.common_availability:

                if time in self.blocked_times or time in used_times:

                    continue

                day = time.split()[0]  # Assume time format is "Monday 10:00"

                if group.school in daily_schools[day]:

                    continue  # Already visiting that school today



                self.schedule[day].append((group.school, time, [s.name for s in group.students]))

                daily_schools[day].add(group.school)

                used_times.add(time)

                break



        return self.schedule



# Sample data

students = [

    Student(1, "Alice", "School A", "articulation", ["Monday 10:00", "Tuesday 11:00"]),

    Student(2, "Bob", "School A", "articulation", ["Monday 10:00", "Monday 11:00"]),

    Student(3, "Charlie", "School A", "articulation", ["Monday 10:00"]),

    Student(4, "Daisy", "School B", "fluency", ["Monday 10:00"]),

    Student(5, "Eve", "School B", "fluency", ["Monday 10:00", "Tuesday 10:00"]),

]



blocked_times = ["Monday 11:00"]



scheduler = Scheduler(students, blocked_times)

schedule = scheduler.generate_schedule()



import pandas as pd

df = pd.DataFrame([

    {"Day": day, "School": school, "Time": time, "Students": ", ".join(students)}

    for day, events in schedule.items()

    for school, time, students in events

])


df

from IPython.display import display
display(df)