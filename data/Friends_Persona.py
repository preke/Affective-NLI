import pandas as pd
df = pd.read_csv('friends-personality.csv')

# df['character'] = df['character'].apply(lambda x: x.split()[0] if (x != 'Phoebe Sr.') else x)
df['raw_text'] = df['raw_text'].apply(lambda x: [[i.split('</b>:')[0].replace('<b>', ''), i.split('</b>:')[1]] for i in x.split('<br>') if "</b>:" in i ])


# use main role or not...

# main_roles = ['Chandler', 'Monica', 'Phoebe', 'Ross', 'Rachel', 'Joey']
# df = df[df['character'].isin(main_roles)]


traits = ['cAGR','cCON','cEXT','cOPN','cNEU']
df_labels = df[traits]
df_counts = pd.DataFrame([])
for trait in traits:
    df_counts[trait] = df_labels[trait].value_counts().reset_index().sort_index(ascending=True)['count']


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
MAX_LEN   = 128


def get_text_role(sent_list, role):
    '''
    Extract the utterances from the given role
    '''
    ans = ""
    for i in sent_list:
        ## if i[0].split(' ')[0] == role and i[0] != 'Phoebe Sr.':
        if i[0] == role:
            ans = ans + ' ' + i[1]
    return ans

def get_context_role(sent_list, role):
    '''
    Extract the utterances not from the given role
    '''
    ans = ""
    for i in sent_list:
        # if i[0].split(' ')[0] != role or i[0] == 'Phoebe Sr.':
        if i[0] != role:
            ans = ans + ' ' + i[1]
    return ans

def get_seg_id(sent_list, role):
    '''
    Generate the segment id for the whole sent
    '''

    ans = []
   
    for i in sent_list:
        if i[0].split(' ')[0] != role.split(' ')[0]:
            
            ans.append(0)
        else:
            ans.append(1)


    return ans

def get_sent(sent_list, role):
    '''
    Obtain the whole sent
    '''
    
    ans = ""
    for i in sent_list:
        ans = ans + i[1]
    return ans

df_A = df[['scene_id', 'raw_text', 'characters', 'cAGR']]
df_C = df[['scene_id', 'raw_text', 'characters', 'cCON']]
df_E = df[['scene_id', 'raw_text', 'characters', 'cEXT']]
df_O = df[['scene_id', 'raw_text', 'characters', 'cOPN']]
df_N = df[['scene_id', 'raw_text', 'characters', 'cNEU']]

df_A['dialog_state'] = df_A.apply(lambda r: get_seg_id(r['raw_text'], r['characters']), axis=1)
df_C['dialog_state'] = df_C.apply(lambda r: get_seg_id(r['raw_text'], r['characters']), axis=1)
df_E['dialog_state'] = df_E.apply(lambda r: get_seg_id(r['raw_text'], r['characters']), axis=1)
df_O['dialog_state'] = df_O.apply(lambda r: get_seg_id(r['raw_text'], r['characters']), axis=1)
df_N['dialog_state'] = df_N.apply(lambda r: get_seg_id(r['raw_text'], r['characters']), axis=1)

df_A['sent'] = df_A.apply(lambda r: get_sent(r['raw_text'], r['characters']), axis=1)
df_C['sent'] = df_C.apply(lambda r: get_sent(r['raw_text'], r['characters']), axis=1)
df_E['sent'] = df_E.apply(lambda r: get_sent(r['raw_text'], r['characters']), axis=1)
df_O['sent'] = df_O.apply(lambda r: get_sent(r['raw_text'], r['characters']), axis=1)
df_N['sent'] = df_N.apply(lambda r: get_sent(r['raw_text'], r['characters']), axis=1)

df_A['utterance'] = df_A.apply(lambda r: get_text_role(r['raw_text'], r['characters']), axis=1)
df_C['utterance'] = df_C.apply(lambda r: get_text_role(r['raw_text'], r['characters']), axis=1)
df_E['utterance'] = df_E.apply(lambda r: get_text_role(r['raw_text'], r['characters']), axis=1)
df_O['utterance'] = df_O.apply(lambda r: get_text_role(r['raw_text'], r['characters']), axis=1)
df_N['utterance'] = df_N.apply(lambda r: get_text_role(r['raw_text'], r['characters']), axis=1)

df_A['context'] = df_A.apply(lambda r: get_context_role(r['raw_text'], r['characters']), axis=1)
df_C['context'] = df_C.apply(lambda r: get_context_role(r['raw_text'], r['characters']), axis=1)
df_E['context'] = df_E.apply(lambda r: get_context_role(r['raw_text'], r['characters']), axis=1)
df_O['context'] = df_O.apply(lambda r: get_context_role(r['raw_text'], r['characters']), axis=1)
df_N['context'] = df_N.apply(lambda r: get_context_role(r['raw_text'], r['characters']), axis=1)

df_A['labels'] = df_A['cAGR'].apply(lambda x: 1 if x is True else 0)
df_C['labels'] = df_C['cCON'].apply(lambda x: 1 if x is True else 0)
df_E['labels'] = df_E['cEXT'].apply(lambda x: 1 if x is True else 0)
df_O['labels'] = df_O['cOPN'].apply(lambda x: 1 if x is True else 0)
df_N['labels'] = df_N['cNEU'].apply(lambda x: 1 if x is True else 0)


df_A.to_csv('Friends_A_whole.tsv', sep='\t', index=False) 
df_C.to_csv('Friends_C_whole.tsv', sep='\t', index=False)
df_E.to_csv('Friends_E_whole.tsv', sep='\t', index=False)
df_O.to_csv('Friends_O_whole.tsv', sep='\t', index=False)
df_N.to_csv('Friends_N_whole.tsv', sep='\t', index=False)




