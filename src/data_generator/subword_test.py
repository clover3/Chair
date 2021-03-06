from data_generator.subword_translate import move_cursor


def main():
    word_tokens = ['Schools', 'should', 'better', 'respect', 'children�s', 'rights', 'Editorial', ',', 'The', 'Korea',
                   'Times', ',', 'August', '22', ',', '2010', 'South', 'Koreans', 'have', 'long', 'taken', 'it', 'for',
                   'granted', 'that', 'teachers', 'have', 'the', 'right', 'to', 'inflict', 'corporal', 'punishment',
                   'against', 'unruly', 'or', 'disobedient', 'students', '.', 'Such', 'punishment', 'has', 'even',
                   'been', 'dubbed', 'a', 'teachers�', '``', 'stick', 'of', 'love�', 'for', 'students', '.', 'This',
                   'has', 'also', 'reflected', 'parents�', 'aspirations', 'for', 'the', 'proper', 'education', 'of',
                   'their', 'children', '.', 'But', 'now', ',', 'physical', 'punishment', 'can', 'no', 'longer', 'be',
                   'tolerated', 'as', 'it', 'is', 'under', 'attack', 'for', 'infringing', 'on', 'student�s', 'basic',
                   'human', 'rights', '.', 'On', 'Monday', ',', 'the', 'Seoul', 'Metropolitan', 'Office', 'of',
                   'Education', 'decided', 'to', 'ban', 'any', 'type', 'of', 'physical', 'punishment', 'at', 'schools',
                   'under', 'its', 'jurisdiction', 'from', 'next', 'semester', '.', 'The', 'move', 'came', 'after', 'a',
                   'primary', 'school', 'teacher', 'was', 'suspended', 'from', 'his', 'job', 'last', 'week', 'for',
                   'having', 'repeatedly', 'beaten', 'his', 'pupils', '.', 'The', 'case', 'shocked', 'the', 'nation',
                   'as', 'major', 'TV', 'channels', 'aired', 'a', 'video', 'clip', 'showing', 'the', '52-year-old',
                   'teacher', ',', 'identified', 'as', 'Oh', ',', 'slapping', ',', 'shoving', 'and', 'kicking', 'sixth',
                   'graders', 'in', 'a', 'classroom', 'in', 'a', 'Seoul', 'elementary', 'school', '.', 'In', 'every',
                   'respect', ',', 'Oh', 'went', 'too', 'far', 'in', 'wielding', 'the', '�stick', 'of', 'love�', 'for',
                   'his', 'students', '.', 'Unquestionably', ',', 'most', 'viewers', 'believe', 'his', 'acts',
                   'constitute', 'sheer', 'violence', 'against', 'children', '.', 'His', 'use', 'of', 'corporal',
                   'punishment', 'can', 'not', 'be', 'justified', 'under', 'any', 'circumstances', '.', 'Needless',
                   'to', 'say', ',', 'violence', 'is', 'not', 'the', 'thing', 'to', 'be', 'mixed', 'with', 'education',
                   '.', 'In', 'this', 'regard', ',', 'many', 'parents', 'welcome', 'the', 'plan', 'to', 'prohibit',
                   'physical', 'punishment', 'in', 'all', 'schools', 'and', 'kindergartens', '.', 'But', 'the', 'total',
                   'ban', 'is', 'touching', 'off', 'a', 'heated', 'debate', 'with', 'conservative', 'teachers', 'and',
                   'school', 'officials', 'opposing', 'it', '.', 'Of', 'course', ',', 'the', 'municipal', 'education',
                   'office', 'has', 'made', 'a', 'hasty', 'decision', 'without', 'reflecting', 'different', 'opinions',
                   'from', 'all', 'walks', 'of', 'life', '.', 'Under', 'the', 'leadership', 'of', 'the',
                   'newly-elected', 'liberal', 'superintendent', 'Kwak', 'No-hyun', ',', 'the', 'office', 'must',
                   'have', 'put', 'a', 'stress', 'on', 'the', 'rights', 'of', 'students', '.', 'It', 'has', 'begun',
                   'to', 'show', 'a', 'set', 'of', 'policy', 'changes', 'from', 'its', 'previous', 'focus', 'on',
                   'market-oriented', 'competition', 'in', 'education', '.', 'Expectations', 'are', 'growing', 'that',
                   'Kwak', 'will', 'push', 'for', 'reform', 'to', 'normalize', 'school', 'education', '.', 'In', 'this',
                   'context', ',', 'the', 'office', 'should', 'have', 'made', 'more', 'efforts', 'to', 'build',
                   'public', 'consensus', 'on', 'the', 'ban', '.', 'The', 'office', 'must', 'work', 'together', 'with',
                   'the', 'education', 'ministry', ',', 'teachers�', 'unions', 'and', 'associations', ',', 'and',
                   'parents�', 'groups', 'to', 'humbly', 'listen', 'to', 'the', 'pros', 'and', 'cons', 'of', 'the',
                   'issue', '.', 'Besides', ',', 'it', 'should', 'not', 'waste', 'their', 'time', 'and', 'efforts',
                   'to', 'persuade', 'opponents', 'of', 'the', 'ban', 'to', 'accept', 'its', 'aim', 'to', 'make',
                   'schools', 'violence-free', '.', 'The', 'opponents', 'also', 'have', 'to', 'realize', 'that', 'all',
                   'education', 'activities', 'should', 'be', 'conducted', 'through', 'legitimate', 'and', 'peaceful',
                   'means', 'in', 'order', 'to', 'achieve', 'the', 'purpose', 'of', 'education', '.', 'No', 'doubt',
                   'corporal', 'punishment', 'has', 'little', 'educational', 'effect', 'on', 'children', ',',
                   'especially', 'when', 'teachers', 'abuse', 'the', '``', 'stick', 'of', 'love�', 'to', 'vent',
                   'their', 'anger', 'on', 'their', 'students', '.', 'Under', 'the', 'current', 'education', 'law', ',',
                   'physical', 'punishment', 'is', 'in', 'principle', 'outlawed', '.', 'It', 'only', 'allows',
                   'teachers', 'to', 'use', 'corporal', 'punishment', 'in', 'exceptional', 'circumstances', 'and', '``',
                   'only', 'for', 'educational', 'purposes.�', 'In', 'this', 'sense', ',', 'no', 'one', 'can', 'deny',
                   'that', 'teachers', 'have', 'long', 'abused', 'the', 'rules', 'to', 'control', 'students', 'and',
                   'maintain', 'order', 'in', 'schools', '.', 'Now', ',', 'teachers', 'and', 'schools', 'should',
                   'change', 'themselves', 'to', 'minimize', 'the', 'abuse', 'and', 'maximize', 'educational',
                   'effects', 'without', 'sacrificing', 'the', 'basic', 'rights', 'of', 'children', '.', 'HAVE', 'YOU',
                   'BEEN']
    subword_tokens = ['schools', 'should', 'better', 'respect', 'children', '##s', 'rights', 'editorial', ',', 'the',
                      'korea', 'times', ',', 'august', '22', ',', '2010', 'south', 'koreans', 'have', 'long', 'taken',
                      'it', 'for', 'granted', 'that', 'teachers', 'have', 'the', 'right', 'to', 'in', '##flict',
                      'corporal', 'punishment', 'against', 'un', '##ru', '##ly', 'or', 'di', '##so', '##bed', '##ient',
                      'students', '.', 'such', 'punishment', 'has', 'even', 'been', 'dubbed', 'a', 'teachers', '`', '`',
                      'stick', 'of', 'love', 'for', 'students', '.', 'this', 'has', 'also', 'reflected', 'parents',
                      'aspirations', 'for', 'the', 'proper', 'education', 'of', 'their', 'children', '.', 'but', 'now',
                      ',', 'physical', 'punishment', 'can', 'no', 'longer', 'be', 'tolerated', 'as', 'it', 'is',
                      'under', 'attack', 'for', 'in', '##fr', '##inging', 'on', 'students', 'basic', 'human', 'rights',
                      '.', 'on', 'monday', ',', 'the', 'seoul', 'metropolitan', 'office', 'of', 'education', 'decided',
                      'to', 'ban', 'any', 'type', 'of', 'physical', 'punishment', 'at', 'schools', 'under', 'its',
                      'jurisdiction', 'from', 'next', 'semester', '.', 'the', 'move', 'came', 'after', 'a', 'primary',
                      'school', 'teacher', 'was', 'suspended', 'from', 'his', 'job', 'last', 'week', 'for', 'having',
                      'repeatedly', 'beaten', 'his', 'pupils', '.', 'the', 'case', 'shocked', 'the', 'nation', 'as',
                      'major', 'tv', 'channels', 'aired', 'a', 'video', 'clip', 'showing', 'the', '52', '-', 'year',
                      '-', 'old', 'teacher', ',', 'identified', 'as', 'oh', ',', 'slapping', ',', 'shoving', 'and',
                      'kicking', 'sixth', 'graders', 'in', 'a', 'classroom', 'in', 'a', 'seoul', 'elementary', 'school',
                      '.', 'in', 'every', 'respect', ',', 'oh', 'went', 'too', 'far', 'in', 'wielding', 'the', 'stick',
                      'of', 'love', 'for', 'his', 'students', '.', 'un', '##quest', '##ion', '##ably', ',', 'most',
                      'viewers', 'believe', 'his', 'acts', 'constitute', 'sheer', 'violence', 'against', 'children',
                      '.', 'his', 'use', 'of', 'corporal', 'punishment', 'cannot', 'be', 'justified', 'under', 'any',
                      'circumstances', '.', 'needles', '##s', 'to', 'say', ',', 'violence', 'is', 'not', 'the', 'thing',
                      'to', 'be', 'mixed', 'with', 'education', '.', 'in', 'this', 'regard', ',', 'many', 'parents',
                      'welcome', 'the', 'plan', 'to', 'prohibit', 'physical', 'punishment', 'in', 'all', 'schools',
                      'and', 'kindergarten', '##s', '.', 'but', 'the', 'total', 'ban', 'is', 'touching', 'off', 'a',
                      'heated', 'debate', 'with', 'conservative', 'teachers', 'and', 'school', 'officials', 'opposing',
                      'it', '.', 'of', 'course', ',', 'the', 'municipal', 'education', 'office', 'has', 'made', 'a',
                      'hasty', 'decision', 'without', 'reflecting', 'different', 'opinions', 'from', 'all', 'walks',
                      'of', 'life', '.', 'under', 'the', 'leadership', 'of', 'the', 'newly', '-', 'elected', 'liberal',
                      'superintendent', 'kw', '##ak', 'no', '-', 'hyun', ',', 'the', 'office', 'must', 'have', 'put',
                      'a', 'stress', 'on', 'the', 'rights', 'of', 'students', '.', 'it', 'has', 'begun', 'to', 'show',
                      'a', 'set', 'of', 'policy', 'changes', 'from', 'its', 'previous', 'focus', 'on', 'market', '-',
                      'oriented', 'competition', 'in', 'education', '.', 'expectations', 'are', 'growing', 'that', 'kw',
                      '##ak', 'will', 'push', 'for', 'reform', 'to', 'normal', '##ize', 'school', 'education', '.',
                      'in', 'this', 'context', ',', 'the', 'office', 'should', 'have', 'made', 'more', 'efforts', 'to',
                      'build', 'public', 'consensus', 'on', 'the', 'ban', '.', 'the', 'office', 'must', 'work',
                      'together', 'with', 'the', 'education', 'ministry', ',', 'teachers', 'unions', 'and',
                      'associations', ',', 'and', 'parents', 'groups', 'to', 'hum', '##bly', 'listen', 'to', 'the',
                      'pro', '##s', 'and', 'con', '##s', 'of', 'the', 'issue', '.', 'besides', ',', 'it', 'should',
                      'not', 'waste', 'their', 'time', 'and', 'efforts', 'to', 'persuade', 'opponents', 'of', 'the',
                      'ban', 'to', 'accept', 'its', 'aim', 'to', 'make', 'schools', 'violence', '-', 'free', '.', 'the',
                      'opponents', 'also', 'have', 'to', 'realize', 'that', 'all', 'education', 'activities', 'should',
                      'be', 'conducted', 'through', 'legitimate', 'and', 'peaceful', 'means', 'in', 'order', 'to',
                      'achieve', 'the', 'purpose', 'of', 'education', '.', 'no', 'doubt', 'corporal', 'punishment',
                      'has', 'little', 'educational', 'effect', 'on', 'children', ',', 'especially', 'when', 'teachers',
                      'abuse', 'the', '`', '`', 'stick', 'of', 'love', 'to', 'vent', 'their', 'anger', 'on', 'their',
                      'students', '.', 'under', 'the', 'current', 'education', 'law', ',', 'physical', 'punishment',
                      'is', 'in', 'principle', 'outlawed', '.', 'it', 'only', 'allows', 'teachers', 'to', 'use',
                      'corporal', 'punishment', 'in', 'exceptional', 'circumstances', 'and', '`', '`', 'only', 'for',
                      'educational', 'purposes', '.', 'in', 'this', 'sense', ',', 'no', 'one', 'can', 'deny', 'that',
                      'teachers', 'have', 'long', 'abused', 'the', 'rules', 'to', 'control', 'students', 'and',
                      'maintain', 'order', 'in', 'schools', '.', 'now', ',', 'teachers', 'and', 'schools', 'should',
                      'change', 'themselves', 'to', 'minimize', 'the', 'abuse', 'and', 'maximize', 'educational',
                      'effects', 'without', 'sac', '##ri', '##fi', '##cing', 'the', 'basic', 'rights', 'of', 'children',
                      '.', 'have', 'you', 'been']
    tokens = ['"', 'judges', 'can', 'google', 'your', 'evidence', ',', 'so', 'make', 'sure', 'it', '’', 's', 'good',
              '|', 'main', '|', 'don', '’', 't', 'spam', 'your', 'presiding', 'judge', '»', 'when', 'the', 'punishment',
              'doesn', '’', 't', 'fit', 'the', 'crime', 'i', 'never', 'thought', 'i', 'would', 'write', 'another',
              'law', 'blog', 'about', 'dvd', 'rentals', '.', 'although', 'under', 'entirely', 'different',
              'circumstances', '(', 'netflix', '“', 'outing', '”', 'their', 'customers', ')', ',', 'the', 'most',
              'interesting', 'legal', 'issues', 'can', 'sometimes', 'be', 'found', 'in', 'the', 'funniest', 'places',
              '.', 'case', 'and', 'point', ':', 'a', ',', 'colorado', 'judge', 'issued', 'an', 'arrest', 'warrant',
              'for', 'a', 'local', 'teenager', 'who', 'has', '$', '30', 'in', 'overdue', 'dvd', '’', 's', 'from', 'the',
              'local', 'library', '.', 'when', 'the', 'teen', 'recently', 'got', 'pulled', 'over', 'for', 'a',
              'routine', 'traffic', 'violation', ',', 'he', 'was', 'placed', 'in', 'jail', 'for', '8', 'hours', 'in',
              'accordance', 'with', 'his', 'arrest', 'warrant', '.', 'the', 'best', 'part', 'is', 'that', 'the',
              'overdue', 'dvds', 'had', 'actually', 'been', 'returned', 'a', 'week', 'before', 'the', 'arrest', '!',
              'call', 'me', 'crazy', ',', 'but', 'overdue', 'dvd', '’', 's', 'should', 'not', 'have', 'jail', 'time',
              'as', 'the', 'punishment', '.', 'i', 'am', 'pretty', 'sure', 'the', 'fine', 'itself', 'is', 'the',
              'punishment', ',', 'but', 'that', '’', 's', 'just', 'me', '.', 'perhaps', 'the', 'colorado', 'judge',
              'read', 'about', 'the', '12', 'year', 'old', 'new', 'york', 'student', 'who', 'was', 'taken', 'from',
              'her', 'classroom', 'in', 'handcuffs', 'for', 'doodling', 'her', 'name', 'on', 'her', 'desk', 'in',
              'erasable', 'markers', '.', 'yes', ',', 'erasable', 'markers', '.', 'growing', 'up', ',', 'kids', 'learn',
              'that', 'they', 'are', 'punished', 'when', 'they', 'do', 'something', 'wrong', '.', 'and', 'when', 'they',
              'do', 'something', 'really', 'bad', ',', 'they', 'get', 'in', 'bigger', 'trouble', '.', 'this',
              'simplistic', 'philosophy', 'guides', 'the', 'legal', 'system', 'for', 'punishment', 'of', 'those',
              'individuals', 'that', 'break', 'the', 'law', '.', 'petty', 'theft', 'carries', 'a', 'much', 'lesser',
              'punishment', 'than', 'a', 'murder', 'charge', '.', 'and', 'rightfully', 'so', '.', 'there', 'are', 'a',
              'lot', 'of', 'sentencing', 'options', 'depending', 'on', 'the', 'crime', 'committed', '.', 'a', 'recent',
              'legalmatch', 'study', 'revealed', 'an', 'interest', 'in', 'changing', 'and', 'lessening', 'punishments',
              'and', 'an', 'increasing', 'interest', 'in', 'alternative', 'sentencing', '.', 'traditionally', ',',
              'when', 'an', 'individual', 'gets', 'convicted', 'of', 'a', 'crime', ',', 'the', 'judge', 'has', 'the',
              'option', 'to', 'impose', 'fines', ',', 'jail', 'time', ',', 'probation', ',', 'or', 'a', 'combination',
              'thereof', '.', 'with', 'alternative', 'sentencing', ',', 'an', 'individual', 'may', 'have', 'to', 'do',
              'community', 'service', ',', 'be', 'placed', 'on', 'house', 'arrest', ',', 'or', 'some', 'form', 'of',
              'work', 'release', '.', 'although', 'i', 'don', '’', 't', 'believe', 'any', 'type', 'of', 'sentencing',
              'is', 'really', 'necessary', 'for', 'dvd', '’', 's', 'or', 'doodling', ',', 'alternative', 'sentencing',
              'is', 'a', 'much', 'more', 'rehabilitative', 'approach', 'to', 'crime', 'and', 'punishment', 'and',
              'offers', 'an', 'opportunity', 'to', 'shape', 'the', 'punishment', 'to', 'the', 'individual',
              'personality', 'of', 'the', 'offender', '.', 'placing', 'individuals', '(', 'teenagers', 'and',
              'children', 'nonetheless', ')', 'under', 'arrest', 'for', 'minor', 'infractions', 'is', 'absolutely',
              'ridiculous', 'and', 'an', 'embarrassment', 'to', 'the', 'legal', 'system', 'as', 'a', 'whole', '.', 'no',
              'system', 'is', 'perfect', 'but', 'egregious', 'punishments', 'like', 'the', 'two', 'examples', 'above',
              'should', 'not', 'happen', '.', 'by', ':', 'violet', 'petran']
    subword_tokens_to_pass = ['«', 'judges', 'can', 'google', 'your', 'evidence', ',', 'so', 'make', 'sure', 'it', '’',
                              's', 'good', '|', 'main', '|', 'don', '’', 't', 'spa', '##m', 'your', 'presiding',
                              'judge', '»', 'when', 'the', 'punishment', 'doesn', '’', 't', 'fit', 'the', 'crime', 'i',
                              'never', 'thought', 'i', 'would', 'write', 'another', 'law', 'blog', 'about', 'dvd',
                              'rental', '##s', '.', 'although', 'under', 'entirely', 'different', 'circumstances', '(',
                              'netflix', '“', 'outing', '”', 'their', 'customers', ')', ',', 'the', 'most',
                              'interesting', 'legal', 'issues', 'can', 'sometimes', 'be', 'found', 'in', 'the', 'fun',
                              '##nies', '##t', 'places', '.', 'case', 'and', 'point', ':', 'a', ',', 'colorado',
                              'judge', 'issued', 'an', 'arrest', 'warrant', 'for', 'a', 'local', 'teenager', 'who',
                              'has', '$', '30', 'in', 'over', '##due', 'dvd', '’', 's', 'from', 'the', 'local',
                              'library', '.', 'when', 'the', 'teen', 'recently', 'got', 'pulled', 'over', 'for', 'a',
                              'routine', 'traffic', 'violation', ',', 'he', 'was', 'placed', 'in', 'jail', 'for', '8',
                              'hours', 'in', 'accordance', 'with', 'his', 'arrest', 'warrant', '.', 'the', 'best',
                              'part', 'is', 'that', 'the', 'over', '##due', 'dvds', 'had', 'actually', 'been',
                              'returned', 'a', 'week', 'before', 'the', 'arrest', '!', 'call', 'me', 'crazy', ',',
                              'but', 'over', '##due', 'dvd', '’', 's', 'should', 'not', 'have', 'jail', 'time', 'as',
                              'the', 'punishment', '.', 'i', 'am', 'pretty', 'sure', 'the', 'fine', 'itself', 'is',
                              'the', 'punishment', ',', 'but', 'that', '’', 's', 'just', 'me', '.', 'perhaps', 'the',
                              'colorado', 'judge', 'read', 'about', 'the', '12', 'year', 'old', 'new', 'york',
                              'student', 'who', 'was', 'taken', 'from', 'her', 'classroom', 'in', 'handcuffs', 'for',
                              'doo', '##dling', 'her', 'name', 'on', 'her', 'desk', 'in', 'eras', '##able', 'markers',
                              '.', 'yes', ',', 'eras', '##able', 'markers', '.', 'growing', 'up', ',', 'kids', 'learn',
                              'that', 'they', 'are', 'punished', 'when', 'they', 'do', 'something', 'wrong', '.', 'and',
                              'when', 'they', 'do', 'something', 'really', 'bad', ',', 'they', 'get', 'in', 'bigger',
                              'trouble', '.', 'this', 'sim', '##pl', '##istic', 'philosophy', 'guides', 'the', 'legal',
                              'system', 'for', 'punishment', 'of', 'those', 'individuals', 'that', 'break', 'the',
                              'law', '.', 'petty', 'theft', 'carries', 'a', 'much', 'lesser', 'punishment', 'than', 'a',
                              'murder', 'charge', '.', 'and', 'rightful', '##ly', 'so', '.', 'there', 'are', 'a', 'lot',
                              'of', 'sentencing', 'options', 'depending', 'on', 'the', 'crime', 'committed', '.', 'a',
                              'recent', 'legal', '##mat', '##ch', 'study', 'revealed', 'an', 'interest', 'in',
                              'changing', 'and', 'less', '##ening', 'punishments', 'and', 'an', 'increasing',
                              'interest', 'in', 'alternative', 'sentencing', '.', 'traditionally', ',', 'when', 'an',
                              'individual', 'gets', 'convicted', 'of', 'a', 'crime', ',', 'the', 'judge', 'has', 'the',
                              'option', 'to', 'impose', 'fines', ',', 'jail', 'time', ',', 'probation', ',', 'or', 'a',
                              'combination', 'thereof', '.', 'with', 'alternative', 'sentencing', ',', 'an',
                              'individual', 'may', 'have', 'to', 'do', 'community', 'service', ',', 'be', 'placed',
                              'on', 'house', 'arrest', ',', 'or', 'some', 'form', 'of', 'work', 'release', '.',
                              'although', 'i', 'don', '’', 't', 'believe', 'any', 'type', 'of', 'sentencing', 'is',
                              'really', 'necessary', 'for', 'dvd', '’', 's', 'or', 'doo', '##dling', ',', 'alternative',
                              'sentencing', 'is', 'a', 'much', 'more', 'rehab', '##ili', '##tative', 'approach', 'to',
                              'crime', 'and', 'punishment', 'and', 'offers', 'an', 'opportunity', 'to', 'shape', 'the',
                              'punishment', 'to', 'the', 'individual', 'personality', 'of', 'the', 'offender', '.',
                              'placing', 'individuals', '(', 'teenagers', 'and', 'children', 'nonetheless', ')',
                              'under', 'arrest', 'for', 'minor', 'in', '##fra', '##ctions', 'is', 'absolutely',
                              'ridiculous', 'and', 'an', 'embarrassment', 'to', 'the', 'legal', 'system', 'as', 'a',
                              'whole', '.', 'no', 'system', 'is', 'perfect', 'but', 'e', '##gre', '##gio', '##us',
                              'punishments', 'like', 'the', 'two', 'examples', 'above', 'should', 'not', 'happen', '.',
                              'by', ':', 'violet', 'petra', '##n']

    # tokens = word_tokens
    # subword_tokens_to_pass = subword_tokens
    target_idx = 100
    move_cursor(0, subword_tokens_to_pass, 0, tokens, target_idx, True)

if __name__ == "__main__":
    main()