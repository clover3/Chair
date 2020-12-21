import requests

from contradiction.medical_claims.ncbi_api_key import api_key


def e_fetch(pmid) -> bytes:
    print(pmid)
    url_format = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?' \
    'db=pubmed' \
    '&id={}' \
    '&api_key={}' \
    '&rettype=xml'

    url = url_format.format(pmid, api_key)
    print(url)
    res = requests.get(url)
    if res.status_code == 200:
        return res.content
    else:
        print(res.status_code)
        print(res.content)
        raise Exception()


if __name__ == "__main__":
    pmid = "23435582"
    print(type(e_fetch(pmid)))
