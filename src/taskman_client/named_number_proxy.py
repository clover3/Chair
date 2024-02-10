from taskman_client.RESTProxy import RESTProxy
from taskman_client.host_defs import webtool_host, webtool_port


class NamedNumberProxy:
    def __init__(self):
        self.server_proxy = RESTProxy(webtool_host, webtool_port)

    def search_numbers(self, keyword):
        return self.server_proxy.get(f"/experiment/named_number/?search={keyword}")
