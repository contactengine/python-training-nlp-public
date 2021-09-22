#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data processing script to turn the JSON format intent classification data
into a CSV file ready for ML model use.
"""

import json
import pandas as pd

# read the raw json data
with open('./oos_data/data_oos_plus.json', 'r') as f:
    data = json.load(f)

# transform it into a dataframe
frames = {}
for key in data.keys():
    df = pd.DataFrame(data[key])
    df.columns = pd.Index(['text', 'intent'])
    df.index.name = 'index'
    frames[key] = df.assign(key=key)

df = pd.concat(frames.values()).reset_index().drop(columns='index')
print('Number of utterances loaded for each split:')
print(df.key.value_counts(), end='\n\n')

# split the intents into categories (using information from the paper appendix)
intents_by_category = {}
intents_by_category['banking'] = {
    'transfer',
    'transactions',
    'balance',
    'freeze_account',
    'pay_bill',
    'bill_balance',
    'bill_due',
    'interest_rate',
    'routing',
    'min_payment',
    'order_checks',
    'pin_change',
    'report_fraud',
    'account_blocked',
    'spending_history',
}
intents_by_category['credit_card'] = {
    'credit_score',
    'report_lost_card',
    'credit_limit',
    'rewards_balance',
    'new_card',
    'application_status',
    'card_declined',
    'international_fees',
    'apr',
    'redeem_rewards',
    'credit_limit_change',
    'damaged_card',
    'replacement_card_duration',
    'improve_credit_score',
    'expiration_date',
}
intents_by_category['small_talk'] = {
    'greeting',
    'goodbye',
    'tell_joke',
    'where_are_you_from',
    'how_old_are_you',
    'what_is_your_name',
    'who_made_you',
    'thank_you',
    'what_can_i_ask_you',
    'what_are_your_hobbies',
    'do_you_have_pets',
    'are_you_a_bot',
    'meaning_of_life',
    'who_do_you_work_for',
    'fun_fact',
}
intents_by_category['oos'] = {'oos'}

# assign categories to the data
df['category'] = pd.NA

for category, intents in intents_by_category.items():
    df['category'] = df.apply(
        lambda x: category if x.intent in intents else x.category, axis=1
    )

print('Categories selected:')
print(df.category.value_counts(), end='\n\n')
print('Number of utterances within these categories:')
print(len(df.dropna()), end='\n\n')
print('Breakdown of intents with these categories:')
print(df.dropna().intent.value_counts(), end='\n\n')

# rename the split in order to no longer separate train and test
# (putting oos train into test and oos test into train because of the
# relative numbers of each of those; the dataset itself has artificially
# small numbers of oos in the training set otherwise)

df['split'] = df.key.str.replace('oos_train', 'test')\
    .str.replace('oos_test', 'train')\
    .str.replace('oos_val', 'val')

# drop other categories (the NAs), shuffle and output the data
out = df.drop(columns='key').dropna().sample(frac=1)  
out.to_csv('data.csv', index=False)
