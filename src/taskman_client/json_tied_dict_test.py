from taskman_client.sync import JsonTiedDict


def main():
    json_tied = JsonTiedDict("tied.json")
    print("last_executed_task_id", json_tied.last_id())
    json_tied.set("last_executed_task_id", 4)
    print("last_executed_task_id", json_tied.last_id())


if __name__ == "__main__":
    main()