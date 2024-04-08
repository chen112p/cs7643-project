import pandas as pd
import os
import numpy as np


def calculate_percentage(column):
    return column.sum() / column.count()

def process_unhealthy_conv(full_data, threshold = [0,0.5]):
    """
    out of antagonize,condescending,dismissive,generalisation,generalisation_unfair,healthy,hostile,sarcastic labels,
    i selected antagonize,condescending,dismissive,generalisation,sarcastic and hostile as "offensive"
    If equal or less than threshold[0] (default 0%) of the rater think a comment is offensive, then it is non-offensive;
    If equal or more than threshold[1] (default 50%) of the rater think a comment is offensive, then it is offensive;
    Comment that are neither considered as offensive or non-offensive are removed (considered as unsure)
    Junting Chen
    """
    grouped_summary = full_data.groupby(['_unit_id','comment']).agg(antagonize_percentage=('antagonize', calculate_percentage),
                    hostile_percentage=('hostile', calculate_percentage),
                    condescending_percentage=('condescending', calculate_percentage),
                    dismissive_percentage=('dismissive', calculate_percentage),
                    generalisation_percentage=('generalisation', calculate_percentage),
                    sarcastic_percentage=('sarcastic', calculate_percentage),
                    counts=('_unit_id', 'size')).reset_index()

    grouped_offensive_or_not = grouped_summary[['_unit_id','comment']].copy()
    grouped_offensive_or_not['offensive'] = (grouped_summary['antagonize_percentage'] >= threshold[1]) | \
                                                (grouped_summary['hostile_percentage'] >= threshold[1]) | \
                                                (grouped_summary['condescending_percentage'] >= threshold[1]) | \
                                                (grouped_summary['dismissive_percentage'] >= threshold[1]) | \
                                                (grouped_summary['generalisation_percentage'] >= threshold[1]) | \
                                                (grouped_summary['sarcastic_percentage'] >= threshold[1])
    grouped_offensive_or_not['nonoffensive'] = (grouped_summary['antagonize_percentage'] <= threshold[0]) & \
                                                (grouped_summary['hostile_percentage'] <= threshold[0]) & \
                                                (grouped_summary['condescending_percentage'] <= threshold[0]) & \
                                                (grouped_summary['dismissive_percentage'] <= threshold[0]) & \
                                                (grouped_summary['generalisation_percentage'] <= threshold[0]) & \
                                                (grouped_summary['sarcastic_percentage'] <= threshold[0])

    grouped_offensive_or_not = grouped_offensive_or_not[grouped_offensive_or_not[['offensive','nonoffensive']].sum(axis=1)!=0].reset_index(drop=True)
    grouped_offensive_or_not = grouped_offensive_or_not.drop('nonoffensive', axis=1)
    print('Percentage of offensive: {:.3f}'.format(grouped_offensive_or_not['offensive'].sum() / grouped_offensive_or_not.shape[0] * 100))
    print('Percentage of non-offensive: {:.3f}'.format(100 - grouped_offensive_or_not['offensive'].sum() / grouped_offensive_or_not.shape[0] * 100))
    print('Total number of rows in the output: {}'.format(grouped_offensive_or_not.shape[0]))
    return(grouped_offensive_or_not)
def process_hate_speech_and_offensive_lang(full_data):
    """
    if the label is hateful or offensive, in our work its labeled as offensive

    Juinting Chen
    """
    full_data['id'] = np.arange(0,full_data.shape[0])
    offensive_or_not = full_data[['id','tweet','class']].copy()
    offensive_or_not['offensive'] = offensive_or_not['class'].isin([0, 1])
    offensive_or_not = offensive_or_not.drop('class',axis=1)
    print('Percentage of offensive: {:.3f}'.format(offensive_or_not['offensive'].sum() / offensive_or_not.shape[0] * 100))
    print('Percentage of non-offensive: {:.3f}'.format(100 - offensive_or_not['offensive'].sum() / offensive_or_not.shape[0] * 100))
    print('Total number of rows in the output: {}'.format(offensive_or_not.shape[0]))
    return(offensive_or_not)