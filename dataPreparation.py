from collections import defaultdict


class DataPreparation:

    def __init__(self):
        pass

    def prepare_data_frame(self, df, train):
        # Uklanjanje duplikata utakmica
        index_to_drop = df[df["teamLoc"] == "Away"].index
        df.drop(index_to_drop, inplace=True)
        df.reset_index(inplace=True)
        df.drop(["index"], axis=1, inplace=True)

        # Promena naziva kolona
        df.rename(columns={'team': 'homeTeam', 'opponent': 'awayTeam'}, inplace=True)

        # Dodavanje "homeWin" kolone --> label
        df["homeWin"] = df["teamRslt"] == "Win"

        # Procenat pobeda domacina
        n_games = df["homeWin"].count()
        n_homewins = df["homeWin"].sum()
        win_percentage = n_homewins / n_games
        if train:
            print("Home Win percentage in train seasons (from data): {0: .2f}%".format(100 * win_percentage))
        else:
            print("Home Win percentage in test seasons (from data): {0: .2f}%".format(100 * win_percentage))

        # Dodavanje "ptsDifference" kolone --> label
        df["ptsDifference"] = df["teamPTS"] - df["opptPTS"]

        self.last_match_result(df)
        self.streak(df)
        # self.last_year_standings(df)
        self.last_between_two(df)

    # Postoji samo da ne trazi da pravim staticke metode
    def is_not_used(self):
        pass

    def last_match_result(self, df):
        self.is_not_used()

        # Rečnik za čuvanje rezultata utakmice koju je tim poslednju odigrao
        won_last = {
            "ATL": False, "BOS": False, "BKN": False, "CHA": False, "CHI": False, "CLE": False,
            "DAL": False, "DEN": False, "DET": False, "GS": False, "HOU": False, "IND": False,
            "LAC": False, "LAL": False, "MEM": False, "MIA": False, "MIL": False, "MIN": False,
            "NO": False, "NY": False, "OKC": False, "ORL": False, "PHI": False, "PHO": False,
            "POR": False, "SAC": False, "SA": False, "TOR": False, "UTA": False, "WAS": False
        }

        df["homeTeamLastWin"] = False
        df["awayTeamLastWin"] = False

        for index_team, row_df in df.iterrows():
            team_h = row_df["homeTeam"]
            team_a = row_df["awayTeam"]
            row_df["homeTeamLastWin"] = won_last[team_h]
            row_df["awayTeamLastWin"] = won_last[team_a]
            df.iloc[index_team] = row_df
            if row_df["teamPTS"] > row_df["opptPTS"]:
                won_last[team_h] = True
            else:
                won_last[team_h] = False

            won_last[team_a] = not won_last[team_h]

    def streak(self, df):
        self.is_not_used()

        df["homeTeamWinStreak"] = 0
        df["awayTeamWinStreak"] = 0

        win_streak = {
            "ATL": 0, "BOS": 0, "BKN": 0, "CHA": 0, "CHI": 0, "CLE": 0,
            "DAL": 0, "DEN": 0, "DET": 0, "GS": 0, "HOU": 0, "IND": 0,
            "LAC": 0, "LAL": 0, "MEM": 0, "MIA": 0, "MIL": 0, "MIN": 0,
            "NO": 0, "NY": 0, "OKC": 0, "ORL": 0, "PHI": 0, "PHO": 0,
            "POR": 0, "SAC": 0, "SA": 0, "TOR": 0, "UTA": 0, "WAS": 0
        }

        for index, row in df.iterrows():
            team = row["homeTeam"]
            opponent = row["awayTeam"]
            row["homeTeamWinStreak"] = win_streak[team]
            row["awayTeamWinStreak"] = win_streak[opponent]
            df.iloc[index] = row
            if row["teamPTS"] > row["opptPTS"]:
                win_streak[team] += 1
                win_streak[opponent] = 0
            else:
                win_streak[opponent] += 1
                win_streak[team] = 0

    def last_between_two(self, df):
        self.is_not_used()

        last_match_winner = defaultdict(int)

        def home_team_won_last(row):
            team = row["homeTeam"]
            opponent = row["awayTeam"]
            teams = tuple(sorted([team, opponent]))
            result = 1 if last_match_winner[teams] == row["homeTeam"] else 0
            winner = row["homeTeam"] if row["homeWin"] else row["awayTeam"]
            last_match_winner[teams] = winner
            return result

        df["homeTeamWonLast"] = df.apply(home_team_won_last, axis=1)
