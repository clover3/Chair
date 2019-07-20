from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc.client


PORT_DOCREADER = 8125


class TextReaderServer:
    def __init__(self, text_reading_fn):
        self.doc_dict = text_reading_fn()


    def start(self, port = PORT_DOCREADER):
        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = ('/RPC2',)

        print("Preparing server")
        server = SimpleXMLRPCServer(("ingham.cs.umass.edu", port),
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
    def __init__(self, port = PORT_DOCREADER):
        self.server = xmlrpc.client.ServerProxy('http://ingham.cs.umass.edu:{}'.format(port))

    def retrieve(self, doc_id):
        return self.server.retrieve(doc_id)



dummy_doc = """Language: <F P=105> English </F>
Article Type:CSO 

<F P=106> [Article by Stanislav Yevgenyev: "The Establishment of </F>
Interstate Bank Marks the First Step Toward an Economic Union"; 
first paragraph is introductory paragraph] 
  [Text] Bankers from national banks of the CIS countries met 
in Moscow to discuss the future structure of a financial 
community. 
  The Central Bank of Russia held a three-day seminar with the 
participation of the International Monetary Fund and the 
European Political Research Centre in order to organize an 
interstate bank. 
  The interstate bank is meant to be a supranational structure 
for effecting settlements between the CIS countries for state 
deliveries. In the course of discussions the participants which 
included representatives of central banks of the Commonwealth 
countries became distinctly aware that the interstate bank will 
either never get off the ground or will require participant 
countries to reform their financial systems. 
  The decision on the formation of the interstate bank with 
its 
headquarters in Moscow was discussed for several years and was 
made last autumn. All members of the future bank agreed that its 
main function would be to deal with interstate trade. The 
present system of state payments between the CIS countries does 
not suit most of them since sums for exported goods are entered 
to accounts of enterprises of the CIS countries in Russian 
commercial banks. (The exporters of the "near abroad" countries 
regard the ruble as a relatively hard currency.) Bankers believe 
that the flight from national currencies to the rouble does not 
help stabilize other soft currencies, encourages the leakage of 
capital and causes the shortfall in taxes. State agencies are 
interested therefore in changing the order of interstate 
settlements. 
  Ironically, the seminar was prepared by specialists of the 
International Monetary Fund and conducted by the Central Bank of 
Russia. The interstate bank still has no office, equipment and 
staff. (It is true that it has received a contribution from 
Russia of 2.5 billion roubles, which is half of the total. The 
contributions of other participants vary depending on their 
share in interstate trade.) But this arrangement is quite 
justified since the choice of the bank's model will determine 
the methods of solving several problems of principle which are 
not directly connected with its activity. For example, the 
experts of the International Monetary Fund discovered on the 
very first day of the seminar that, although the colleagues from 
the CIS countries showed interest in the subject, they were on 
the whole sceptical. It became clear from subsequent talks that 
their doubts stemmed not from the bank's model but from the 
conditions in which it will have to be implemented. The 
interstate bank can be considered stillborn if a single 
settlement mechanism is applied to countries with different 
currency, monetary and licence rules. The participants in the 
meeting became distinctly aware of the necessity of the 
unification of several conditions, which is, alas, beyond their 
competence. 
  Russia, which has a positive trade balance, can painlessly 
change over to a new system of interstate settlements. The 
situation of Turkmenistan is also quite satisfactory. The other 
states have debts to repay for which they will need credits. 
Sources of credit are needed and debts must be formalized. 
Specialists from national banks admit that countries find it 
difficult today to separate the debts of states from those of 
individuals. Thus most countries will be in a quandary because 
of the changeover to the new system which does not allow 
arbitrary crediting. When the debt exceeds the established limit 
the account of the participant country in the interstate bank 
can be blocked, which means that payments will be frozen and 
trade will stop. The debtor state will then have to sell part of 
its hard currency reserves in order to cover the debt balance. 
The changeover to a system of settlements through the interstate 
bank will also mean for most CIS countries the automatic 
introduction of compulsory 100 per cent sale of proceeds in hard 
currency since clearing settlements between countries will be 
done in roubles and with the final clients in their own national 
currency. (It may be noted for comparison that Kazakhstan has 
until recently practised 30 per cent compulsory sale of hard 
currency. The exporter kept 70 per cent of the total, which was 
often kept abroad.) The deportation of exporters' money home, 
especially in national currency, can hardly be welcomed by them. 
  The establishment of the interstate bank should, according 
to 
the intention of the contracting countries of the former USSR 
conclude the first stage of a new economic union. During the 
second stage they plan to form a payments union for countries 
which cannot satisfy the current conditions to enter the 
interstate bank. Settlements between the participants in the 
payments union will be also effected in roubles which will be 
considered the reserve currency. 
  The uncoordinated market rules in the various former union 
republics make it difficult to establish a payments system 
similar to the European one. It is entirely unclear now, at the 
first stage what goods will be entered in the list of state 
deliveries, whether they will be licensed, whether the system of 
settlements through correspondent accounts in commercial banks, 
practised until now, will be preserved, and how rates of 
exchange will be coordinated. The European payments system 
presupposes, for instance, a mechanism to regulate exchange 
rates (introduction of a fixed rate of exchange) with a 
permissible deviation of plus-minus one. But in order to 
introduce such a system all states participating in the union 
should possess adequate currency reserves for supporting the 
fixed rate of exchange by means of currency interventions in the 
market. Most CIS countries have insufficient reserves of hard 
currency. If the rate of exchange is established freely, through 
a currency exchange or an auction, it should cover all deals 
concluded in the country and not be divided into official and 
black-market rates. It should not be forgotten, above all, that 
the elaboration of a currency policy is a state matter and that 
banks are totally powerless to change the rules of the game. 
  In any case the establishment of an interstate bank would 
shorten the time required for payments and their guarantee. 
  At the same time the interstate bank will not work if the 
rules of the game in the countries participating in the union 
remain uncoordinated. Since suppliers find it better to live 
"under the old regime" it will evidently be necessary to 
introduce bureaucratic checks as well. 
"""

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