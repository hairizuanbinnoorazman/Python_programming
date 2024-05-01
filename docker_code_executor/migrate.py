import sqlite3

con = sqlite3.connect("example.db")
cur = con.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS job(
    id TEXT PRIMARY KEY, 
    status TEXT NOT NULL, 
    date_created TEXT NOT NULL,
    date_modified TEXT NOT NULL)""")
con.commit()