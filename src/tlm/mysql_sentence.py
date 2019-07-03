import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="1234",
)

cursor = mydb.cursor()
cursor.execute("use robust;")

g_id = 0
def add_sents(doc_id, sents):
  sql = "INSERT INTO RobustSents (DocId, Sentence, loc, g_id) VALUES (%s, %s, %s, %s)"
  vals = []
  global g_id
  for loc, sent in enumerate(sents):
    g_id += 1
    vals.append((doc_id, sent, loc, g_id))
  cursor.executemany(sql, vals)
  mydb.commit()


def get_sent(idx):
  sql = "SELECT * from RobustSents where s_id=" + str(idx)
  cursor.execute(sql)
  row = cursor.fetchone()
  return row





