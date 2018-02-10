import math
import operator
import pandas as pd
import numpy as np

from simulation_package.exponential import Exponential
from simulation_package.GeoCoordSimulator import GeoCoordSimulator


class SFFireSimulation:
    def __init__(self, n_server, days, sf_fires_df):
        self.N_servers = n_server
        self.simulation_days = days
        self.df = sf_fires_df

    @staticmethod
    def timefseconds(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return int(d)+1, int(h), int(m), int(s)

    @staticmethod
    def find_first_empty_server(SS):
        for i in range(0, len(SS)):
            if SS[i] == 0:
                return i

    @staticmethod
    def get_min_time_server(times):
        return min(enumerate(times), key=operator.itemgetter(1))

    #VALUTAARE PERMUTAZIONI AL POSTO CHE RAND
    def simulate(self):
        #init utility variables
        fires_exp = Exponential(1 / 1050.73914786)
        server_service_exp = Exponential(1 / 1050.7392199)
        primary = self.df['Action Taken Primary'].value_counts(normalize=True)
        secondary = self.df['Action Taken Secondary'].value_counts(normalize=True)
        other = self.df['Action Taken Other'].value_counts(normalize=True)
        location_df = self.df[['Longitude', 'Latitude']].dropna()
        gcs = GeoCoordSimulator(location_df)
        gcs.calculate_clusters()
        # init variables
        # Time variable
        t = 0
        # System State variables (SS)
        n_live_fires = 0
        SS = [0 for _ in range(self.N_servers)]
        # Counter variables
        Na = 0
        C = [0 for _ in range(self.N_servers)]
        # output variables
        simulated_fire = pd.DataFrame(columns=['Latitude',
                                               'Longitude',
                                               'Action Taken Primary',
                                               'Action Taken Secondary',
                                               'Action Taken Other'])
        A = []
        D = []
        # Event list
        Tserver = [math.inf for _ in range(self.N_servers)]

        Ta = fires_exp.random()
        while t < self.simulation_days*86400:
            if Ta <= min(Ta, *Tserver):
                # the next events is a fire
                t = Ta
                Na += 1
                Ta = t + fires_exp.random()

                primary_action = np.random.choice(primary.index, 1, p=primary)
                secondary_action = np.random.choice(secondary.index, 1, p=secondary)
                other_actions = np.random.choice(other.index, 1, p=other)
                coord = gcs.get_random_coordinates(1)

                simulated_fire = simulated_fire.append({'Latitude': coord[0,0],
                                                        'Longitude': coord[0,1],
                                                        'Action Taken Primary': primary_action[0],
                                                        'Action Taken Secondary': secondary_action[0],
                                                        'Action Taken Other': other_actions[0]}, ignore_index=True)
                A.append(t)
                n_live_fires += 1
                if n_live_fires <= self.N_servers:
                    i = self.find_first_empty_server(SS)
                    SS[i] = Na
                    Tserver[i] = t + server_service_exp.random()
            else:
                # the next events is an intervention end
                min_time_ix, min_time = self.get_min_time_server(Tserver)
                t = Tserver[min_time_ix]
                C[min_time_ix] += 1
                D.append(t)
                if n_live_fires <= self.N_servers:
                    SS[min_time_ix] = 0
                    Tserver[min_time_ix] = math.inf
                else:
                    SS[min_time_ix] += 1
                    Tserver[min_time_ix] = t + server_service_exp.random()
                n_live_fires -= 1

        A_hour = ["{} day {:d}:{:d}:{:d}".format(*self.timefseconds(sec)) for sec in A]
        D_hour = ["{} day {:d}:{:d}:{:d}".format(*self.timefseconds(sec)) for sec in D]
        simulated_fire = simulated_fire[0:len(D_hour)]
        simulated_fire['(A) Alarm DateTime'] = A_hour[0:len(D_hour)]
        simulated_fire['(D) Resolution DateTime'] = D_hour
        simulated_fire['A (in sec)'] = A[0:len(D_hour)]
        simulated_fire['D (in sec)'] = D
        simulated_fire['diff'] = simulated_fire['D (in sec)'] - simulated_fire['A (in sec)']
        return simulated_fire

    def simulate_only_times(self):
        # init utility variables
        fires_exp = Exponential(1 / 1050.73914786)
        server_service_exp = Exponential(1 / 1050.7392199)
        # init variables
        # Time variable
        t = 0
        # System State variables (SS)
        n_live_fires = 0
        SS = [0 for _ in range(self.N_servers)]
        # Counter variables
        Na = 0
        C = [0 for _ in range(self.N_servers)]
        A = []
        D = []
        # Event list
        Tserver = [math.inf for _ in range(self.N_servers)]

        Ta = fires_exp.random()
        while t < self.simulation_days * 86400:
            if Ta <= min(Ta, *Tserver):
                # the next events is a fire
                t = Ta
                Na += 1
                Ta = t + fires_exp.random()
                A.append(t)
                n_live_fires += 1
                if n_live_fires <= self.N_servers:
                    i = self.find_first_empty_server(SS)
                    SS[i] = Na
                    Tserver[i] = t + server_service_exp.random()
            else:
                # the next events is an intervention end
                min_time_ix, min_time = self.get_min_time_server(Tserver)
                t = Tserver[min_time_ix]
                C[min_time_ix] += 1
                D.append(t)
                if n_live_fires <= self.N_servers:
                    SS[min_time_ix] = 0
                    Tserver[min_time_ix] = math.inf
                else:
                    SS[min_time_ix] += 1
                    Tserver[min_time_ix] = t + server_service_exp.random()
                n_live_fires -= 1

        A = A[0:len(D)]

        return np.array(D)-np.array(A)
