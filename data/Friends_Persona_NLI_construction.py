import pandas as pd
import os
from sklearn.model_selection import train_test_split


SEED_list = [0,42,3407,1,2,3,4,5,6,7]


def process(r):
    text_list = r['raw_text']
    speaker = r['characters']
    ans = ''
    for text in text_list:
        if text[0] == speaker:
            ans += 'Speaker' + ' : ' + text[1] + ', '
        else:
            ans += 'Others' + ' : ' + text[1] + ', '
    ans += '; '
    return ans

def process_only_speaker(row):
    text_list = row['raw_text']
    speaker = row['characters']
    
    ans = ''
    for text in text_list:
        if text[0] == speaker:
            ans +=  text[1] + ', '
    ans += '; '
    return ans

def get_emotion_prompt(r):
    dialog_state = r['dialog_state']
    emotions = eval(r['Dialog_EmoBERTa_label'])
    character = r['characters'].split()[0]
    ans = ''
    for i in range(len(dialog_state)):
        if i == 0:
            if dialog_state[i] == 0:
                ans += 'First, the emotion of others is ' + emotions[i] + ', '
            elif dialog_state[i] == 1:
                ans += 'The emotion of ' + character + ' is initially ' + emotions[i] + ', '
        else:
            if dialog_state[i] == 0:
                ans += 'the emotion of others is ' + emotions[i] + ', '
            elif dialog_state[i] == 1:
                ans += character + ' respond with ' + emotions[i] + ', '
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

def get_personailty_description(r, p, p_or_n):
        character = r['characters'].split()[0]
        label = r['labels']
        key = p + p_or_n

        return character + ' ' + personality_description[key]

    
    

for flow_len in [0.25, 0.5, 0.75, 1]:

    df = pd.read_csv('friends-personality.csv')
    df['raw_text'] = df['raw_text'].apply(lambda x: [[i.split('</b>:')[0].replace('<b>', ''), i.split('</b>:')[1]] for i in x.split('<br>') if "</b>:" in i ])
    df['raw_text'] = df['raw_text'].apply(lambda x: x[:int(len(x)*flow_len)])

    df_with_emo = pd.read_csv('Friends_A_whole.tsv', sep='\t')

    df['origin_sent'] = df.apply(process, axis=1)
    df['only_speaker'] = df.apply(process_only_speaker, axis=1)


    df['Dialog_EmoBERTa_label'] =  df_with_emo['Dialog_EmoBERTa_label']

    df['dialog_state'] = df.apply(lambda r: get_seg_id(r['raw_text'], r['characters']), axis=1)
    df['affective_prompt'] = df.apply(get_emotion_prompt, axis=1)
    df['sent_and_prompt'] = df['origin_sent'] + df['affective_prompt']

    df_A = df[['scene_id', 'raw_text', 'characters', 'affective_prompt', 'only_speaker', 'origin_sent', 'sent_and_prompt', 'cAGR']]
    df_C = df[['scene_id', 'raw_text', 'characters', 'affective_prompt', 'only_speaker', 'origin_sent', 'sent_and_prompt', 'cCON']]
    df_E = df[['scene_id', 'raw_text', 'characters', 'affective_prompt', 'only_speaker', 'origin_sent', 'sent_and_prompt', 'cEXT']]
    df_O = df[['scene_id', 'raw_text', 'characters', 'affective_prompt', 'only_speaker', 'origin_sent', 'sent_and_prompt', 'cOPN']]
    df_N = df[['scene_id', 'raw_text', 'characters', 'affective_prompt', 'only_speaker', 'origin_sent', 'sent_and_prompt', 'cNEU']]

    df_A['labels'] = df_A['cAGR'].apply(lambda x: 1 if x is True else 0)
    df_C['labels'] = df_C['cCON'].apply(lambda x: 1 if x is True else 0)
    df_E['labels'] = df_E['cEXT'].apply(lambda x: 1 if x is True else 0)
    df_O['labels'] = df_O['cOPN'].apply(lambda x: 1 if x is True else 0)
    df_N['labels'] = df_N['cNEU'].apply(lambda x: 1 if x is True else 0)

    df_A.to_csv('Friends_A_with_role.tsv', sep='\t', index=False) 
    df_C.to_csv('Friends_C_with_role.tsv', sep='\t', index=False)
    df_E.to_csv('Friends_E_with_role.tsv', sep='\t', index=False)
    df_O.to_csv('Friends_O_with_role.tsv', sep='\t', index=False)
    df_N.to_csv('Friends_N_with_role.tsv', sep='\t', index=False)


    personality_description = {
        'A_pos': 'is friendly, cooperative, empathetic, and compassionate, often prioritizing harmonious relationships and the well-being of others.',
        'A_neg': 'is confrontational, uncooperative, lacking empathy, and often prioritizing their own needs and desires over the well-being of others.',
        'C_pos': 'is organized, responsible, diligent, detail-oriented, and committed to achieving their goals with a strong sense of duty and self-discipline.',
        'C_neg': 'is disorganized, careless, impulsive, lacking discipline, and often displaying a disregard for responsibilities and commitments.',
        'E_pos': 'is outgoing, sociable, energetic, and thriving in social interactions, often seeking stimulation and enjoying the company of others.',
        'E_neg': 'is introverted, reserved, quiet, and often preferring solitude or smaller social settings, conserving energy and finding fulfillment in introspection and reflection.',
        'O_pos': 'has curiosity, open-mindedness, creativity, tolerance, emotional expressiveness, and willingness to embrace new experiences and ideas.',
        'O_neg': 'is closed-minded, resistant to change, lacking curiosity, intolerant of differences, emotionally guarded, and hesitant to explore new ideas or experiences.',
        'N_pos': 'is prone to experiencing negative emotions, such as anxiety, worry, and mood swings, often displaying heightened sensitivity to stress and a tendency towards self-doubt and emotional instability.',
        'N_neg': 'is emotionally stable, resilient, and composed, often displaying a calm and balanced demeanor, and having a tendency to handle stress and adversity with ease.'
    }

    


    personality = ['A', 'C', 'E', 'O', 'N']
    for p in personality:
        print(p, '...')
        df_tmp = pd.read_csv('Friends_'+p+'_with_role.tsv', sep='\t')
        
        
        df_tmp['pos_personality_description'] = df_tmp.apply(get_personailty_description, p=p, p_or_n='_pos', axis=1)
        df_tmp['neg_personality_description'] = df_tmp.apply(get_personailty_description, p=p, p_or_n='_neg', axis=1)
        
        df_tmp['personality_description'] = df_tmp['pos_personality_description']
        
        df_tmp['label'] = df_tmp['labels']
        df_tmp['nli_label'] = df_tmp['labels']
        
        df_tmp['sent'] = df_tmp['origin_sent']
        
        df_tmp['affective_dialog'] = df_tmp['sent_and_prompt']
        
        df_tmp['sentence1'] = df_tmp['sent_and_prompt']
        df_tmp['sentence2'] = df_tmp['personality_description']
        
        
        for seed in SEED_list:
            train_df, test_df = train_test_split(df_tmp, test_size=0.2, random_state=seed)
            valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=seed)
        
            train_df.to_csv('Friends_'+p+'_with_role_'+str(flow_len)+'_'+str(seed)+'_train.tsv', sep='\t', index=False)
            valid_df.to_csv('Friends_'+p+'_with_role_'+str(flow_len)+'_'+str(seed)+'_valid.tsv', sep='\t', index=False)
            test_df.to_csv('Friends_'+p+'_with_role_'+str(flow_len)+'_'+str(seed)+'_test.tsv', sep='\t', index=False)

        os.remove('Friends_'+p+'_with_role.tsv')
        
        
        
        