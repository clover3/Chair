import mysql.connector



mydb = None
cursor = None

def init_cursor():
  global cursor
  global mydb
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
  if mydb is None:
    init_cursor()
  sql = "SELECT * from RobustSents where s_id=" + str(idx)
  cursor.execute(sql)
  row = cursor.fetchone()
  return row



def get_sent_gid(g_id):
  if mydb is None:
    init_cursor()
  sql = "SELECT * from RobustSents where g_id=" + str(g_id)
  cursor.execute(sql)
  row = cursor.fetchone()
  return row


def get_doc_sent(doc_id):
  if mydb is None:
    init_cursor()
  sql = "SELECT * from RobustSents where DocId=%s"
  cursor.execute(sql, (doc_id,))
  rows = cursor.fetchall()
  return rows



