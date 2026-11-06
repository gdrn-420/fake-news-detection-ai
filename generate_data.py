"""Create a larger fictional teaching dataset for the classroom text-classification demo.
These are invented examples, NOT verified news and NOT suitable for real fact-checking.
"""
from pathlib import Path
import csv
import random

REAL_TEMPLATES = [
    "The school notice says the science exhibition will take place in the auditorium on {day}.",
    "The public library announced revised opening hours for the month of {month}.",
    "The district office published the schedule for the community clean-up programme.",
    "The transport department posted an update about changes to the student bus timetable.",
    "The examination board released a revised timetable on its official website.",
    "The college announced that applications for the scholarship programme are open until {date}.",
    "The weather department issued a bulletin forecasting rainfall in parts of the region.",
    "The museum website lists the opening hours and ticket information for visitors.",
    "The health department published information about a free vaccination awareness camp.",
    "The school circular asks students to submit their project files to their class teachers.",
    "The city council published the minutes of its monthly public meeting.",
    "The university announced the dates for its upcoming entrance examinations.",
    "The railway website posted a notice about a temporary platform change for selected trains.",
    "The sports department released the fixtures for the inter-school tournament.",
    "The public works office announced a scheduled road maintenance programme.",
    "The science centre invited visitors to a public demonstration about renewable energy.",
]

FAKE_TEMPLATES = [
    "A viral message claims that eating one mango makes people invisible for 24 hours.",
    "An anonymous post claims that humans can fly after drinking a special fruit juice.",
    "A forwarded message says a secret button can change every student's exam marks instantly.",
    "A post claims that the moon will disappear tomorrow and gives no evidence or source.",
    "A message promises guaranteed full marks if students share it with {number} friends.",
    "An online rumour says ordinary paper can turn into gold overnight using a secret trick.",
    "A sensational post claims that sleeping for ten minutes replaces an entire year of study.",
    "An unknown account says all school examinations are cancelled but provides no official notice.",
    "A viral claim says people can breathe underwater after eating a special leaf.",
    "A forwarded message promises instant wealth by clicking an unknown link before midnight.",
    "A post claims that a person can stop time by wearing a magnetic bracelet.",
    "An anonymous message says the city will disappear tonight but offers no reliable evidence.",
    "A sensational headline claims that one tablet gives everyone perfect memory in seconds.",
    "A viral post says a secret formula can make rocks grow into full trees overnight.",
    "A message claims that the sun will not rise tomorrow and asks readers to forward it.",
    "An unknown website promises guaranteed exam success without studying or attending classes.",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Friday", "15 November", "next week"]
MONTHS = ["January", "March", "June", "September", "December"]
DATES = ["10 November", "15 December", "the end of this month", "next Friday"]
NUMBERS = ["5", "10", "15", "30"]

def main():
    random.seed(2026)
    rows = []
    for _ in range(240):
        template = random.choice(REAL_TEMPLATES)
        rows.append((template.format(day=random.choice(DAYS), month=random.choice(MONTHS), date=random.choice(DATES)), "REAL"))
    for _ in range(240):
        template = random.choice(FAKE_TEMPLATES)
        rows.append((template.format(number=random.choice(NUMBERS)), "FAKE"))
    random.shuffle(rows)
    out = Path("data/news_samples.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Created {len(rows)} fictional teaching examples at {out}")
    print("Reminder: examples are invented patterns, not verified real-world news.")

if __name__ == "__main__":
    main()
