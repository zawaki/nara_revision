import sqlite3
import numpy as np
import random as rnd
import time

#ALIBABA TEST-TRAIN SPLITTING
conn = sqlite3.connect('''PUT OWN DATABASE PATH HERE''')
cur = conn.cursor()

cur.execute("SELECT machine_id, time_stamp FROM container_meta")

all_containers = cur.fetchall()
l = len(all_containers)
r = np.random.rand(l)

vect = lambda x: 1 if x <= 0.8 else 0
vect_f = np.vectorize(vect)
r = vect_f(r)
r = [x for x in r]

for i in range(l):
    if i % 1000:
        print(i)
    m_id = all_containers[i][0]
    t_s = all_containers[i][1]
    cur.execute("UPDATE container_meta SET train=? WHERE machine_id=? AND time_stamp=?",(int(r[i]),m_id,t_s))

conn.commit()

#AZURE TEST-TRAIN SPLITTING
conn = sqlite3.connect('''PUT OWN DATABASE PATH HERE''')
cur = conn.cursor()

cur.execute("SELECT core FROM vmType",())
l = len(cur.fetchall())

r = np.random.rand(l)
vect = lambda x: 1 if x <= 0.8 else 0
vect_f = np.vectorize(vect)
r = vect_f(r)
r = [x for x in r]
r[0]

for i in range(l):
    cur.execute("UPDATE vmType SET train=? WHERE id=?",(int(r[i]),i))

conn.commit()