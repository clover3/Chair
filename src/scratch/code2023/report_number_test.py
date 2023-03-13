from taskman_client.task_proxy import get_task_manager_proxy


def main():
    data = {'name': 'splade_regresion_2_50000', 'number': 0.53125, 'condition': '',
            'machine': 't1v-n-2be9b43b-w-0',
     'field': 'pairwise_acc'}
    proxy = get_task_manager_proxy()
    proxy.post("/experiment/update", data)

    return NotImplemented


if __name__ == "__main__":
    main()