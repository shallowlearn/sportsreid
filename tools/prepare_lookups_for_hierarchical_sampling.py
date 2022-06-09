'''
A script to prepare all the lookup dicts
needed by the RandomIdentitySampler_Hierarchical
Author:
'''
import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from pprint import pprint
import pickle
def create_lookupdict(list_of_files, data_source=None):

    # Map pid to action
    pid_action = {}
    # Map action to pids
    action_pid = {}
    # Map pid to match
    pid_match = {}
    # Map match to pid
    match_pid = {}
    # Match pid to a pair of teams, year and league combined
    pid_team_year_league = {}
    # Map pair of teams, year and league to a set of pids
    team_year_league_pid = {}
    # Match pid to a pair of teams across years in the same league
    pid_team_league = {}
    # Map a pair of teams across years in the same league to a set of pids
    team_league_pid = {}
    # Map pid to match involving atleast one of the two teams in the same year in the same league
    pid_atleast_one_team_year_league = {}
    # Map match involving atleast one of the two teams in the same year in the same league to set of pids
    atleast_one_team_year_league_pid = {}
    # Map pid to match involving atleast one of the two teams across years in the same league
    pid_atleast_one_team_league = {}
    # Map match involving atleast one of the two teams across years in the same league to set of pids
    atleast_one_team_league_pid = {}

    # For each file, get pid, action, match, year, league, pair of teams involved
    for eachidx, each in enumerate(list_of_files):
        # Split into filename and folder
        folder, fname = os.path.split(each)
        if data_source is None:
            # Get action and pid from file name
            action, pid = fname.split('-')[1:3]
        else:
            # Get action and pid from data_source
            # This is needed if we are using train + test for training
            pid = data_source[eachidx][1]
            action = data_source[eachidx][2]

            # action2, pid2 = fname.split('-')[1:3]
            # assert int(pid2) == pid
            # assert int(action2) == action


        # split foldername at train if it is train folder
        if '/train' in folder:
            _, folder = folder.split('train/')
        elif '/gallery_query' in folder:
            # split foldername at gallery_query if it is gallery_query folder
            # This happens if we are including test images for training
            _, folder = folder.split('gallery_query/')
        # split folder at '/'
        league, year, match, _ = folder.split('/')

        # get teams from match
        splits = re.split('\d ',match)
        team_one = splits[-3].strip() # Remove white spaces
        team_two = splits[-1].strip() # Remove white spaces

        # We will sort team one and two alphabetically so that we get unique keys
        team_one, team_two = sorted((team_one, team_two))

        # Make pid an integer
        pid = int(pid)

        # Map pid to action. One pid can only have one action.
        # Same pid can appear in different images all with the same action
        # indicating that it is the same player
        # Sanity check that a given pid has the exact same action
        safe_update(pid_action, pid, action)

        # Map action to pids. It is a set of all pids that belong to a given action
        safe_update(action_pid, action, pid, isset=True)

        # Map pid to match
        safe_update(pid_match, pid, match)

        # Map match to pid. It is a set
        safe_update(match_pid, match, pid, isset=True)

        # Map pid to same pair of teams in this year in the same league. Maybe a second-leg
        val = '_'.join([team_one, team_two, year, league])
        safe_update(pid_team_year_league, pid, val)

        # Map same pair of teams in this year in the same league to set of pids
        safe_update(team_year_league_pid, val, pid, isset=True)

        # Map pid to same pair of teams in a league across years
        val = '_'.join([team_one, team_two, league])
        safe_update(pid_team_league, pid, val)

        # Map same pair of teams in a league across years to a set of pids
        safe_update(team_league_pid, val, pid, isset=True)

        # Map pid to match involving atleast one of the two teams in the same year in the same league
        # Map match involving atleast one of the two teams in the same year in the same league to set of pids
        for team in [team_one, team_two]:
            val = '_'.join([team, year, league])
            # Map pid to something
            safe_update(pid_atleast_one_team_year_league, pid, val, isset=True)
            # Map something to pid
            safe_update(atleast_one_team_year_league_pid, val, pid, isset=True)

        # Map pid to match involving atleast one of the two teams across years in the same league
        # Map match involving atleast one of the two teams across years in the same league to set of pids
        for team in [team_one, team_two]:
            val = '_'.join([team, league])
            # Map pid to something
            safe_update(pid_atleast_one_team_league, pid, val, isset=True)
            # Map something to pid
            safe_update(atleast_one_team_league_pid, val, pid, isset=True)

    lookupdict = {
        'pid_action': pid_action,
        'action_pid': action_pid,
        'pid_match': pid_match,
        'match_pid': match_pid,
        'pid_team_year_league': pid_team_year_league,
        'team_year_league_pid': team_year_league_pid,
        'pid_team_league': pid_team_league,
        'team_league_pid': team_league_pid,
        'pid_atleast_one_team_year_league': pid_atleast_one_team_year_league,
        'atleast_one_team_year_league_pid': atleast_one_team_year_league_pid,
        'pid_atleast_one_team_league': pid_atleast_one_team_league,
        'atleast_one_team_league_pid': atleast_one_team_league_pid
    }

    return lookupdict

def safe_update(indict, key, val, isset=False):
    '''
    A function to safely update a dict with a new key, val pair
    :param indict:
    :param key:
    :param val:
    :param isset: If dict[key] is a set
    :return:
    '''
    if isset:
        if key not in indict:
            # Create set
            indict[key] = set()
        indict[key].add(val)
    else:
        if key not in indict:
            indict[key] = val
        else:
            # Make sure that new val matches old val
            assert indict[key] == val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--fil', required=True, type=str, default=None, help='path to file containing list of images')
    #parser.add_argument('-', '--', action='store_true', help='')
 
    args = parser.parse_args()

    # Read in file
    with open(args.fil, 'r') as f:
        list_of_files = f.readlines()
    list_of_files = [i.strip('\n') for i in list_of_files]

    lookupdict = create_lookupdict(list_of_files)

    # Save all the lookups
    outfilename = args.fil.replace('.txt', '_lookup.pkl')

    pickle.dump(lookupdict, open(outfilename, 'wb'))