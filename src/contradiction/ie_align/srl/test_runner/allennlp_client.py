import requests


def main():
    url = "http://localhost:8131/predict"
    sent1 = 'This site includes a list of all award winners and a searchable database of Government Executive articles.'
    sent2 = 'The government searched the site.'
    sent1 = "yeah i i think my favorite restaurant is always been the one closest  you know the closest as long as it's it meets the minimum criteria you know of good food"
    sent2 = "My favorite restaurants are always at least a hundred miles away from my house. "
    data_list = [sent1, sent2]
    for sent in data_list:
        data =  {'sentence': sent}
        response = requests.post(url, json=data)
        print(response.json())


if __name__ == "__main__":
    main()