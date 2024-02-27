import openai
from openai import OpenAI

file_path = r"C:\Users\leste\Downloads\c\key.txt"
key = open(file_path, "r").read().strip()
# openai.organization = "org-H88LVGb8C9Zc6OlcieMZgW6e"
client = OpenAI(api_key=key)


def make_job():

    file_path = r"C:\Users\leste\Downloads\c\ft.txt"

    file = client.files.create(
      file=open(file_path, "rb"),
      purpose="fine-tune"
    )
    print(file)

    ret = client.fine_tuning.jobs.create(
        training_file=file.id,
        model="gpt-3.5-turbo",
    )
    print(ret)

# List 10 fine-tuning jobs
# make_job()
ret = client.fine_tuning.jobs.list(limit=10)
print(ret)
for item in ret:
    print(item)
