from tlm.mysql_sentence import get_sent_gid, mydb
import nltk
from misc_lib import TimeEstimator

# create table terms (term_id INT PRIMARY KEY AUTO INCREMENT, term VARCHAR(255));
# create table invindex (term_id INT, doc_id INT, loc INT);

n_sentence = 1000 * 10000

cursor = mydb.cursor(buffered=True)
cursor.execute("use robust;")

def add_terms(tokens):
    sql = "INSERT INTO terms (term) VALUES (%s)"
    vals = []
    for t in tokens:
        vals.append((t,))
    cursor.executemany(sql, vals)
    mydb.commit()

def add_term(term):
    sql = "INSERT INTO terms (term) VALUES (%(term)s)"
    cursor.execute(sql, {'term':term})


def load_all_voca():
    sql = "select term from terms;"
    cursor.execute(sql)
    rows = cursor.fetchall()
    s = set()
    for row in rows:
        s.add(row[0])
    return s

def insert_invindex(doc_id, loc, term):
    sql = ("insert into invindex "
                  "(term, doc_id, loc) "
                  "values (%(term)s, %(doc_id)s, %(loc)s)")

    cursor.execute(sql, {
        'doc_id':doc_id,
        'loc':loc,
        'term':term,
    })


def bulk_insert_invindex(payload):
    sql = ("insert into invindex "
           "(doc_id, loc, term) "
           "values (%s, %s, %s)")

    cursor.executemany(sql, payload)


def get_term_id(term):
    def get_term_id_inner(term):
        sql = "SELECT * from terms where term=%(term)s;"
        cursor.execute(sql, {'term':term})
        row = cursor.fetchone()
        if row is not None:
            return row[0]
        else:
            return row

    row = get_term_id_inner(term)
    if row is None:
        add_term(term)

    return get_term_id_inner(term)


def index_sentence(doc_id, loc, tokens):
    for t in tokens:
        insert_invindex(doc_id, loc, t)




def start_indexing():
    start = 1
    tick = TimeEstimator(n_sentence, sample_size=1500)
    payload = []
    for g_id in range(start, n_sentence):
        r = get_sent_gid(g_id)
        s_id, doc_id, loc, g_id, sent = r
        tokens = nltk.word_tokenize(sent)
        for t in tokens:
            payload.append((doc_id, loc, t))

        tick.tick()

        if g_id % 500 == 0 :
            bulk_insert_invindex(payload)
            payload = []
            mydb.commit()


if __name__ == "__main__":
    start_indexing()