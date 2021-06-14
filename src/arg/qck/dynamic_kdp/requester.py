import os
import pickle
import socketserver
import xmlrpc.client
from typing import List, Tuple
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

from arg.qck.decl import KDP
from list_lib import lmap
from taskman_client.sync import JsonTiedDict

port = 6006


def request_kdp_eval(kdp_list: List[KDP]):
    server = os.environ['kdp_server']
    host = "{}:{}".format(server, port)
    proxy = xmlrpc.client.ServerProxy(host)
    kdp_list_raw = lmap(KDP.getstate, kdp_list)
    job_id = proxy.eval_job(kdp_list_raw)
    return job_id


class KDPEvalServer:
    def __init__(self):
        self.request_dir = os.environ["request_dir"]
        info_path = os.path.join(self.request_dir, "req_job_info.json")
        self.json_tied_dict = JsonTiedDict(info_path)
        self.next_job_id = self.json_tied_dict.last_id()

    def start(self):
        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = ('/RPC2',)

        class RPCThreading(socketserver.ThreadingMixIn, SimpleXMLRPCServer):
            pass

        print("")
        print("  [ KDPEvalServer ]")
        print()
        print("Preparing server")
        server = RPCThreading(("0.0.0.0", port),
                              requestHandler=RequestHandler,
                              allow_none=True)
        server.register_introspection_functions()
        server.register_function(self.eval_job, 'eval_job')
        print("Waiting")
        server.serve_forever()

    def save_request(self, job_id, kdp_list: List[KDP]):
        save_path = os.path.join(self.request_dir, str(job_id))
        temp_save_path = save_path + ".tmp"
        pickle.dump(kdp_list, open(temp_save_path, "wb"))
        os.rename(temp_save_path, save_path)

    def eval_job(self, kdp_list_raw: List[Tuple]):
        kdp_list: List[KDP] = lmap(KDP.from_state, kdp_list_raw)
        job_id = self.next_job_id
        self.save_request(job_id, kdp_list)
        self.next_job_id += 1
        self.json_tied_dict.set('last_executed_task_id', self.next_job_id)
        return job_id


