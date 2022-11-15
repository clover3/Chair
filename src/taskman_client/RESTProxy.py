import json
from queue import Queue
from threading import Thread
from time import sleep

import requests
from requests import ReadTimeout
from urllib3.exceptions import ConnectTimeoutError


class RESTProxy:
    """
    Makes calls to a given host and server given an address. This simplifies any REST calls to a server.
    """
    def __init__(self, host, port):
        """
        Will store the endpoint address to the server being called. And create async thread to take care of any
        async requests to send to the proxy.
        :param host: The host address
        :param port: The port on the host.
        """
        self.host = host
        self.port = port
        self.url_format = "https://{}:{}".format(host, port)
        self.list = Queue()
        self.executor = Thread(target=self._execute, daemon=True, name="Executor-{}-{}".format(host, port))
        self.executor.start()

    def put_to_async_call(self, method, *args):
        """
        Adds an async call, will not take response in to consideration, will retry until the given endpoint responded.
        :param method:  The method call (will be a sub Proxy-class)
        :param args:    The argument to the call.
        """
        m = (method, *args)
        self.list.put(m)

    def _execute(self):
        """
        Will get any async request and will not take response in to consideration, will retry until the given endpoint
        responded.
        """
        while True:
            item = self.list.get()
            is_done = False
            while not is_done:
                is_done = True
                method = getattr(self, item[0])
                try:
                    method(*item[1])
                except ValueError as e:
                    print("Executor got error when sending request to {}: {}".format(item[0], str(e)))
                except (ReadTimeout, requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout, ConnectTimeoutError) as e:
                    is_done = False
                    sleep(5)


    def post(self, url_postfix, data, timeout=5):
        """
        This method simplifies any failed HTTP POST calls by will return a ValueError
        :param url_postfix: The URL.
        :param data:        The POST data.
        :param timeout:     The timeout
        :return:            The response.

        """
        if len(url_postfix) > 0 and url_postfix[0] != '/':
            print("WARNING postfix should start with /")
        url = self.url_format + url_postfix

        response = requests.post(url, data=json.dumps(data), timeout=timeout)
        if response.status_code < 200 or response.status_code > 300:
            if response.status_code == 408:
                raise ValueError("Timed out")
            raise ValueError(response.json())
        elif response.status_code == 204 or response.status_code == 404:
            return []
        else:
            return response.json()


    def get(self, url_postfix, timeout=5):
        """
        This method simplifies any failed HTTP GET calls by will return a ValueError
        :param url_postfix: The url postfix needed for the call.
        :return:            The body of the call assumed to be json-able.
        :raises ValueError  If unsuccessful HTTP call (successful HTTP code = 2xx) will return an error. Creates an
                            exception handling.
        """
        if len(url_postfix) > 0 and url_postfix[0] != '/':
            print("WARNING postfix should start with /")
        url = self.url_format + url_postfix
        #print("get: {}".format(url))
        response = requests.get(url, timeout=timeout)
        if response.status_code < 200 or response.status_code > 300:
            if response.status_code == 408:
                raise Exception("Timed out")
            print("DEBUG rest_protxy error " , response.status_code)
            raise ValueError(response.json())
        elif response.status_code == 204 or response.status_code == 404:
            return []
        else:
            return response.json()


    def get_ownership(self, item_id, replica_id, timeout=5):
        """
        Makes a ownership request to given replica.
        :param item_id:     The item that is requested for ownership.
        :param replica_id:  The replica called (assumed current owner)
        :param timeout:     The max timeout time.
        :return:            The response.
        """
        assert type(item_id) == int
        assert type(replica_id) == int
        url_postfix = "/replication/owner/{}/{}".format(item_id, replica_id)
        return self.get(url_postfix, timeout=timeout)


