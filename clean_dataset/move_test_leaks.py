# move_test_leaks.py
import csv, os, shutil
out = "duplicates/leak_groups"
os.makedirs(out, exist_ok=True)
with open("duplicates/duplicate_report.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        p = r["path"].replace("\\","/")
        if "/test/" in p:
            tgt = os.path.join(out, os.path.basename(p))
            try:
                shutil.move(p, tgt)
            except Exception as e:
                print("fail:", p, e)
print("Moved test copies to", out)
