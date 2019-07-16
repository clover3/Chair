from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc.client


PORT_DOCREADER = 8125


class TextReaderServer:
    def __init__(self, text_reading_fn):
        self.doc_dict = text_reading_fn()


    def start(self):
        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = ('/RPC2',)

        print("Preparing server")
        server = SimpleXMLRPCServer(("ingham.cs.umass.edu", PORT_DOCREADER),
                                    requestHandler=RequestHandler,
                                    allow_none=True,
                                    )
        server.register_introspection_functions()

        server.register_function(self.retrieve, 'retrieve')
        print("Waiting")
        server.serve_forever()


    def retrieve(self, doc_id):
        print(doc_id)
        if doc_id in self.doc_dict:
            return self.doc_dict[doc_id]
        else:
            print("Not found")
            return None


class TextReaderClient:
    def __init__(self):
        self.server = xmlrpc.client.ServerProxy('http://ingham.cs.umass.edu:{}'.format(PORT_DOCREADER))

    def retrieve(self, doc_id):
        return self.server.retrieve(doc_id)




if __name__ == '__main__':
    def dummy_reader():
        doc_dict ={}
        for i in range(10):
            doc_dict[str(i)] = "{} amam".format(i)
        return doc_dict

    if False:
        server = TextReaderServer(dummy_reader)
        server.start()
    else:
        client = TextReaderClient()
        print(client.retrieve("FBIS3-23791"))