import spacy

from dataset_specific.msmarco.common import load_queries

sampled_queries = """
is water considered organic
what happened as a result of the attack on pearl harbor?
what causes brown algae
what is food worksheet
how much do you have to pay to become an american citizen
definition of chromatic colors
sea otters scientific name
how long is rabbinical school reform
what is a z drive
what is a legal medicine doctor
what is cassava leaf
what is impaired insight judgement
types of seed treatments
what is an example of a planned unit development
what causes of body pain
multiple sclerosis symptoms
flower of mexico
how much do generators for house cost
what county is whittier ca in?
what is integrity in ethics
what is atp synthase?
stanford housing cost per semester
states that can charge lodging tax on per diem rates
what basketball player has scored the most points ever
was the revolutionary war the american revolutionary period
chinese zodiac for 1967
what are the causes for an elevated d-dimer?
what is a mrn
foods that lowers blood pressure naturally
what happened to chris soules in court
is hippocampus part of forebrain
what county is virginia's crooked road in
definition of veterinary
the meaning the word purview
tectonic plate movement names
what is great barrier reef
what is a group of muskrats called
what happened to donald trump in dayton ohio
how many forensic science jobs are there
what clotting factors does the liver produce
definition of inner beauty
what happened to grinkov?
what airport do you fly into for puerto vallarta, mexico?
list all 5 type of relations between organisms
is painting your house rental a repair or improvement
richmond ky is in what county
age for plan b pill fda
how can you tell pregnancy
is it tear or tare
in what year did american women get the right to vote?
cost of solar power
what is a leaf scar
what does the name heather mean
is inherited house taxable
are kazoos cancerous
endoskeleton meaning
healthsource poland
what are betel nuts
what is aams
pay rate for a cook
is trump getting impeached
what are the small bones in the spine called
what is cmv on key
how do i cook a leg of lamb in the slow cooker
what is bengaline
"""


def show_queries():
    queries = load_queries("train")
    for qid, query in queries:
        print(query.strip())


def parse_test():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("merge_noun_chunks")
    queries = sampled_queries.split("\n")
    for q in queries:
        parsed = nlp(q)
        l = []
        for t in parsed:
            l.append(t)
        print(tuple(l))



if __name__ == "__main__":
    parse_test()