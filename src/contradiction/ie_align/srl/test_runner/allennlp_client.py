import requests


def main():
    url = "http://localhost:8131/predict"
    sent1 = 'This site includes a list of all award winners and a searchable database of Government Executive articles.'
    sent2 = 'The government searched the site.'
    sent1 = "Supplementation during pregnancy with a medical food containing L-arginine and vitamins reduced the incidence of pre-eclampsia in a population at high risk of the condition."
    sent2 = "L-Arginine load in pregnant women is associated with increased nitric oxide (NO) production and hypotension."
    data_list = [sent1, sent2]
    for sent in data_list:
        data =  {'sentence': sent}
        response = requests.post(url, json=data)
        print(response.json())


if __name__ == "__main__":
    main()