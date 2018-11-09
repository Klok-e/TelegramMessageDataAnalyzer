import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


class Message:
    def __init__(self, datetime: datetime.datetime, text: str, username: str, user_id: int, first_name: str,
                 last_name: str):
        self.datetime = datetime
        self.text = text
        self.username = username
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name

    def __str__(self):
        return self.text


def load_data():
    import json
    f = open("scraped_information.json", "r")
    data = json.load(f)
    f.close()
    return data


def process_data(data: list):
    time_change_date = datetime.datetime(year=2018, month=10, day=28, hour=2)
    for i in range(0, len(data)):
        message = data[i]
        date = datetime.datetime(second=message[0][0], minute=message[0][1], hour=message[0][2],
                                 day=message[0][3], month=message[0][4], year=message[0][5])
        date += datetime.timedelta(hours=3 if date < time_change_date else 2)
        data[i] = Message(date, message[1], message[2], message[3], message[4], message[5])

    # remove None messages
    data = [i for i in data if i.text is not None]

    # remove bot messages
    new_data = []
    for msg in data:
        if msg.username is not None:
            if "PollBot" != msg.username:
                new_data.append(msg)
        else:
            new_data.append(msg)
    data = new_data

    # remove links and numbers
    import re
    for msg in data:
        msg.text = re.sub(r'(http://|https://)\S*', '', msg.text)
        msg.text = re.sub(r'\d*', '', msg.text)
        msg.text = msg.text.lower()

    # remove empty messages
    data = [i for i in data if i.text != ""]

    return data


def get_message_for_each_hour(data):
    hour_messages = {x: list() for x in range(0, 24)}
    for msg in data:
        hour_messages[msg.datetime.hour].append(msg)
    return [len(hour_messages[x]) for x in hour_messages]


def get_message_for_each_hour_for_every_day_of_the_week(data):
    hour_messages = {y: {x: list() for x in range(0, 24)} for y in range(7)}
    for msg in data:
        hour_messages[msg.datetime.weekday()][msg.datetime.hour].append(msg)

    for weekday in hour_messages:
        for hour in hour_messages[weekday]:
            hour_messages[weekday][hour] = len(hour_messages[weekday][hour])

    for day in hour_messages:
        hour_messages[day] = [hour_messages[day][x] for x in hour_messages[day]]
    return hour_messages


def plot_messages_per_hour(data, label: str):
    message_per_minute_hour = get_message_for_each_hour(data)
    np_data = np.concatenate((np.arange(0, 24).reshape(1, 24), np.asarray(message_per_minute_hour).reshape(1, 24)), 0)

    plt.plot(np_data[0], np_data[1], label=label)
    plt.grid(True)


def plot_messages_per_hour_for_week(data):
    days_messages = get_message_for_each_hour_for_every_day_of_the_week(data)

    mx = days_messages[0][0]
    for i in range(7):
        for k in range(24):
            if mx < days_messages[i][k]:
                mx = days_messages[i][k]

    f, ax = plt.subplots(1, 7)
    for i in range(7):
        ax[i].plot([i for i in range(24)], days_messages[i], scaley=False)
        ax[i].set_yticks([i for i in range(0, mx + 50, 50)])
        if i > 0:
            ax[i].get_yaxis().set_visible(False)


def get_msgs_for_user(id: int, data):
    id_to_fetch = id
    msgs = []
    for msg in data:
        if msg.user_id == id_to_fetch:
            msgs.append(msg)
    return msgs


def get_all_users(data):
    all_users = dict()
    for m in data:
        if m.user_id not in all_users:
            all_users.update({m.user_id: (m.username, m.first_name, m.last_name)})
    return all_users


def plot_and_save_24h_msgs_for_every_user(data):
    all_users = get_all_users(data)
    user_ids = list(all_users.keys())
    user_ids.sort()

    id_to_msgs = {u_id: get_msgs_for_user(u_id, data) for u_id in user_ids}
    mx_msgs = max(get_message_for_each_hour(max(id_to_msgs.values(), key=lambda x: len(x))))

    for i in range(len(user_ids)):
        user_id = user_ids[i]
        msgs = id_to_msgs[user_id]

        plot_messages_per_hour(msgs, '{} {}'.format(user_id, str(all_users[user_id])))

        plt.yticks([i for i in range(0, mx_msgs + (mx_msgs // 10), mx_msgs // 10)])

        if i % 8 == 7 or i == len(user_ids) - 1:
            art = plt.legend(loc="right", bbox_to_anchor=(2.1, 0.5))
            plt.savefig('all{}.pdf'.format(i), bbox_inches='tight', additional_artists=[art, ])
            plt.clf()


def plot_and_save_clusters_of_message_text(data):
    transformed = TfidfVectorizer().fit_transform([msg.text for msg in data])

    points = TruncatedSVD().fit_transform(transformed)

    prev = None
    for k in range(1, 15):
        fl = True
        km = KMeans(n_clusters=k)
        km = km.fit(points)
        if prev is None:
            prev = km
            fl = False
        if fl and prev.inertia_ - km.inertia_ < 6:
            break
        prev = km
    clustering = prev.predict(points)

    cluster_to_msg = dict()
    for i in range(len(clustering)):
        if clustering[i] not in cluster_to_msg:
            cluster_to_msg.update({clustering[i]: [data[i]]})
        else:
            cluster_to_msg[clustering[i]].append(data[i])

    cluster_to_points = dict()
    for i in range(len(clustering)):
        if clustering[i] not in cluster_to_points:
            cluster_to_points.update({clustering[i]: [points[i]]})
        else:
            cluster_to_points[clustering[i]].append(points[i])

    for clust in sorted(cluster_to_points.keys()):
        plt.scatter([i[0] for i in cluster_to_points[clust]], [i[1] for i in cluster_to_points[clust]], label=clust,
                    s=2)
    plt.legend()
    plt.savefig("clusters.pdf", bbox_inches='tight')

    for cluster in cluster_to_msg:
        fle = open("cluster{}.txt".format(cluster), "w", encoding="utf-8")
        for msg in cluster_to_msg[cluster]:
            fle.write(msg.text + "\n")
        fle.close()


def main():
    data = process_data(load_data())

    plot_messages_per_hour_for_week(data)

    plt.show()


if __name__ == "__main__":
    main()
