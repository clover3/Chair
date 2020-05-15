import xmlrpc.client

from cpath import data_path, pjoin
from data_generator.tokenizer_wo_tf import EncoderUnitPlain


def test_run():
    voca_path = pjoin(data_path, "bert_voca.txt")

    encoder = EncoderUnitPlain(512, voca_path)
    text = "The preservation of displaced cultures is important in preventing future oppression." \
    + "Notions of cultural superiority virtually always influence displacement and abuse of indigenous cultures." \
    + "For example, when the government of Botswana expelled the Kalahari Bushmen from their land in 2002," \
    + "President Mogai defended his actions by describing the bushmen as stone age creatures." \
    + "This cultural insensitivity, in addition to the incentive of material gains," \
    + "led the Botswani government to violate the tribe's rights." \
    + "By preserving indigenous culture, governments recognize the value of these groups and prevent future hostility."
    text2 = "The government should rely on just legal systems to protect the rights of indigenous people, " \
            "not cultural preservation. A constitution that enumerates rights and a court system that " \
            "scrutinizes government activity is a much more direct and reliable venue of protecting indigenous rights " \
            "than sporadic funding for cultural programs."
    payload = []
    text2_true_true = "Governments also ignore or destroy culture all the time. Historic and significant buildings are built to build infrastructure, traditions are lost through an unwillingness to provide funding to prevent it from dying. When governments protect culture they inevitably protect one over the others. This is nearly always the culture of the majority. Instead it is not for the government to promote or protect any culture over others rather it should be left to private individuals and each cultural group to promote their own culture."

    text2_false_true = "The government should rely on just legal systems to protect the rights of indigenous people, " \
              "not cultural preservation. A constitution that enumerates rights and a court system that " \
              "scrutinizes government activity is a much more direct and reliable venue of protecting indigenous rights " \
              "than sporadic funding for cultural programs."
    text2_false_true2 = "Indigenous populations have no more right to special government treatment than other minority groups. Even indigenous populations did not inhabit their current territory from the dawn of time, and many ethnic groups around the world live where they do because they were pushed out of some other territory hundreds or thousands of years ago. Virtually every ethnic group in the world has been conquered and abused by some other group. Tracing the entirety of human history to determine which group owes reparations to which other group is unproductive; rather, governments should move forward to promote a better standard of living for all citizens."
    text3_1 = "All of society benefits from protection of indigenous culture " \
              "Across the United States, Australia, and Canada, native customs " \
              "are often tied closely to the land. For example, while descendants of " \
              "the Sioux Indians of the American Midwest may no longer hunt buffalo, " \
              "learning about traditional means of hunting, animal use, rituals involving " \
              "the surrounding wildlife, means of ensuring a sustainable food supply, " \
              "and other cultural norms related to the land gives people a greater " \
              "appreciation for the land they now inhabit. Exposure to traditions that have " \
              "been practiced in one's land for thousands of years helps us to appreciate the legacy " \
              "we have inherited. This does not just benefit the direct descendants of those practicing " \
              "these traditions but the whole of society."
    text3_2 = "Protecting indigenous culture is unlikely to have a significant impact on the general " \
              "population. Native groups often live in relative isolation, thereby having little contact " \
              "with people outside the community. Furthermore, antiquated forms of hunting and cultivating " \
              "food that were used over a hundred years ago have little relevance to the modern environment " \
              "in which people live. Learning about these traditions is unlikely to impact the public's " \
              "perception of its environment because the public is unlikely to make an emotional connection " \
              "between these traditions and their modern homes."
    text3_2n = "Protecting indigenous culture is likely to have a significant impact on the general " \
              "population. Native groups often live in relative isolation, thereby having little contact " \
              "with people outside the community. Furthermore, antiquated forms of hunting and cultivating " \
              "food that were used over a hundred years ago have large relevance to the modern environment " \
              "in which people live. Learning about these traditions is likely to impact the public's " \
              "perception of its environment because the public is likely to make an emotional connection " \
              "between these traditions and their modern homes."

    # Against
    text4_a = "In this case we may never do evil ( directly attack and kill a child via abortion ) so that good ( saving the life of the mother ) may result"
    # Favor
    text4_f1 = 'The choice — the only actual choice , in the world as it really is — is between safe , legal abortion and dangerous , illegal abortion .'
    text4_f1n = 'The choice — the only actual choice , in the world as it really is — is between dangerous, legal abortion and safe no abortion .'
    text4_f2 = "But I am glad I do n't live in a society with a \" you had sex , ( or were raped ) it 's your problem now , no matter how much you hate kids \" approach to parenthood "
    text4_f2n = "But I am sad I do n't live in a society with a \" you had sex , ( or were raped ) it 's your problem now , no matter how much you hate kids \" approach to parenthood "

    text4_a2 = "We live in a world where “ post-birth abortion ” , the murder of newborns , is gaining popularity ."
    text4_a2n = "We live in a world where “ post-birth abortion ” , the saving of newborns , is gaining popularity ."

    text5_f = " equality values help 	 to 	 identify 	 the 	 kinds 	 of 	 restrictions 	 on 	 abortion 	 that 	 are 	 unconstitutional 	 under 	 casey 	 ' 	 s 	 undue 	 burden 	 test"
    text5_fn = " equality values help 	 to 	 identify 	 the 	 kinds 	 of 	 dangers 	 on 	 abortion 	 that 	 are 	 unconstitutional 	 under 	 casey 	 ' 	 s 	 undue 	 burden 	 test"

    text5_a = "the 	 only 	 way 	 to 	 justify 	 their 	 loss 	 of 	 moral 	 innocence 	 " \
              "is 	 to 	 lose 	 their 	 intellectual 	 innocence 	 . "

    text5_a2 = " if 	 we 	 measure 	 the 	 beginning 	 of 	 meaningful 	 life 	 by 	 the 	 same 	 benchmark 	 " \
               "we 	 use 	 to 	 measure 	 the 	 end 	 of 	 meaningful 	 life 	 for 	 comatose 	 " \
               "patients 	 , 	 a 	 secular 	 argument 	 can 	 be 	 made 	 that 	 the 	 beginning 	 of 	 " \
               "brain 	 function 	 , 	 which 	 actually 	 precedes 	 viability 	 , 	 " \
               "is 	 more 	 important 	 in 	 determining 	 the 	 beginning 	 of 	 meaningful 	 personhood 	 " \
               "than 	 likelihood 	 of 	 survival 	 outside 	 the 	 mother 	 ' s 	 body 	 . "
    text5_f2 = "  having 	 a 	 baby 	 greatly 	 " \
               "changes 	 the 	 mother 	 ' 	 s 	 life 	 . "
    text5_f2n = "	 having 	 a 	 baby 	 rarely 	 " \
               "changes 	 the 	 mother 	 ' 	 s 	 life 	 . "
    text5_f2n2 = "  losing a 	 baby 	 greatly 	 " \
               "changes 	 the 	 mother 	 ' 	 s 	 life 	 . "

    payload.append(encoder.encode_pair(text2, text2_true_true))
    payload.append(encoder.encode_pair(text2, text2_false_true))
    payload.append(encoder.encode_pair(text2, text2_false_true))
    print(payload)

    port = 8123
    # Example usage :
    proxy = xmlrpc.client.ServerProxy('http://ingham.cs.umass.edu:{}'.format(port))
    r = proxy.predict(payload)
    print(r)


if __name__ == "__main__":
    test_run()