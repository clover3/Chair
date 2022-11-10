import xmlrpc.client


class ServerProxyEx:
    def __init__(self, server_addr, port):
        self.proxy = xmlrpc.client.ServerProxy('http://{}:{}'.format(server_addr, port))

    def send(self, payload):
        if payload:
            r = self.proxy.predict(payload)
            return r
        else:
            return []
