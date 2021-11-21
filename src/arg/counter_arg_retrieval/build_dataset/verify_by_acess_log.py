"""

166.172.63.112 - - [04/Jul/2021:00:07:46 -0400] "GET / HTTP/1.1" 301 166
166.172.63.112 - - [04/Jul/2021:00:08:46 -0400] "-" 408 -
166.172.63.112 - - [04/Jul/2021:00:09:24 -0400] "GET /clueweb/clueweb12-0206wb-77-26892.html HTTP/1.1" 301 166
166.172.63.112 - - [04/Jul/2021:00:10:24 -0400] "-" 408 -
128.119.246.134 - - [04/Jul/2021:00:16:15 -0400] "\x16\x03\x01\x02" 400 226
128.119.246.134 - - [04/Jul/2021:00:16:15 -0400] "\x16\x03\x01\x02" 400 226
"""
import datetime
import re
import sys
from collections import Counter, defaultdict
from typing import List, Dict
from typing import NamedTuple

import pytz

from misc_lib import group_by
from mturk.parse_util import HitResult, parse_mturk_time


# ('172.16.0.3', '25/Sep/2002:14:04:19 +0200', 'GET / HTTP/1.1', '401', '', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.1) Gecko/20020827')


class ParseError(Exception):
    pass

class ApacheLogRaw(NamedTuple):
    ip: str
    time: str
    request: str
    return_code: str
    unknown_field: str


def time_parse():
    s = '02/Jul/2021:21:24:22 -0400'
    time = datetime.datetime.strptime(s, "%d/%b/%Y:%H:%M:%S %z")
    print(time)


class ApacheLogParsed(NamedTuple):
    ip: str
    time: datetime.datetime
    method: str
    url: str
    protocol: str
    unknown_field: str

    def get_url(self):
        return self.url

    @classmethod
    def from_raw_log(cls, log_raw: ApacheLogRaw):
        tokens = log_raw.request.split()
        try:
            method, url, protocol = tokens
        except ValueError:
            raise ParseError
        time = datetime.datetime.strptime(log_raw.time, "%d/%b/%Y:%H:%M:%S %z")
        return ApacheLogParsed(
            log_raw.ip,
            time,
            method,
            url,
            protocol,
            log_raw.unknown_field
        )


def parse_line(line):
    regex = '([(\d\.)]+) - - \[(.*?)\] "(.*?)" (\d+) (.*?)'
    items = re.match(regex, line).groups()
    return ApacheLogRaw(*items)


def load_log(log_path) -> List[ApacheLogParsed]:
    log_list = []
    for line in open(log_path, "r"):
        raw_log = parse_line(line)
        try:
            log = ApacheLogParsed.from_raw_log(raw_log)
            log_list.append(log)
        except ParseError as e:
            pass

    return log_list


def is_time_between(target, st, ed):
    return st <= target <= ed



def load_apache_log():
    apache_log_path = "C:\\Bitnami\\redmine-4.0.4-1\\apache2\logs\\8888.log"
    log_list = load_log(apache_log_path)
    return log_list


def main():
    # time_parse()
    log_list = load_log(sys.argv[1])
    # 04/Jul/2021:02:47:33
    # tz = timezone('UTC-04:00')
    tz = log_list[0].time.tzinfo
    st = datetime.datetime(2021, 7, 4, 0, 0, 33, tzinfo=tz)
    ed = datetime.datetime(2021, 7, 4, 10, 47, 33, tzinfo=tz)
    print(st, ed)
    for log in log_list:
        print(log.time)
        print(is_time_between(log.time, st, ed))

    # for url in url_grouped:
    #     print(url)


def verify_by_time(hit_results: List[HitResult], apache_logs: List[ApacheLogParsed]):
    suspicious_hit = []
    url_grouped_logs = group_by(apache_logs, ApacheLogParsed.get_url)

    def get_url_for_hit(hit):
        doc_id = hit.inputs['doc_id']
        url = "/clueweb/{}.html".format(doc_id)
        return url

    apache_tz = apache_logs[0].time.tzinfo
    ip_worker_matches = Counter()
    url_grouped_hits = group_by(hit_results, get_url_for_hit)
    for url, hits in url_grouped_hits.items():
        output_lines = []
        try:
            logs = url_grouped_logs[url]

            if len(logs) < len(hits):
                print("WARNING less logs than hits")
            suspicious_hit_for_url = []
            for hit in hits:
                start_time = parse_mturk_time(hit.accept_time)
                end_time = parse_mturk_time(hit.submit_time)

                matched_logs = []
                for log_idx, log in enumerate(logs):
                    assert log.time.tzinfo == apache_tz
                    if is_time_between(log.time, start_time, end_time):
                        matched_logs.append((log_idx, log))
                        ip_worker_matches[log.ip, hit.worker_id] += 1
                if not matched_logs:
                    suspicious_hit_for_url.append(hit)
                line = "{} / {} / {}".format(start_time.astimezone(apache_tz),
                                             end_time.astimezone(apache_tz),
                                             hit.worker_id
                                             )
                if not matched_logs:
                    line += " *"
                output_lines.append(line)

            if len(suspicious_hit_for_url) + len(logs) >= len(hits):
                output_lines.append("url={} / {} + {} >= {}".format(url, len(suspicious_hit_for_url), len(logs), len(hits)))
                pass
            else:
                s = "WARNING #hit={}  #logs={}  #no_match_hit={}".format(len(hits), len(logs), len(suspicious_hit_for_url))
                output_lines.append(s)
            if suspicious_hit_for_url and len(logs) < len(hits):
                output_lines.append(url)
                output_lines.append("<logs>")
                for l in logs:
                    output_lines.append("{} {}".format(l.time, l.ip))
                output_lines.append("<hits>")
            suspicious_hit.extend(suspicious_hit_for_url)
        except KeyError:
            suspicious_hit.extend(hits)

    return suspicious_hit, ip_worker_matches


def verify_by_ip(hit_results: List[HitResult], apache_logs: List[ApacheLogParsed], ip_worker_time_match):
    url_grouped_logs: Dict[str, List[ApacheLogParsed]] = group_by(apache_logs, ApacheLogParsed.get_url)

    def get_url_for_hit(hit):
        doc_id = hit.inputs['doc_id']
        url = "/clueweb/{}.html".format(doc_id)
        return url

    raw_worker_count = Counter([h.worker_id for h in hit_results])
    url_grouped_hits = group_by(hit_results, get_url_for_hit)

    worker_hit_count = Counter()
    ip_count = Counter()
    per_ip_count = defaultdict(Counter)
    suspicious_urls = []
    for url, hits in url_grouped_hits.items():
        try:
            logs: List[ApacheLogParsed] = url_grouped_logs[url]

            if len(logs) < len(hits):
                suspicious_urls.append(url)

            ips = set([log.ip for log in logs])

            ip_count.update(ips)

            for hit in hits:
                worker_hit_count[hit.worker_id] += 1
                for ip in ips:
                    per_ip_count[ip][hit.worker_id] += 1

        except KeyError:
            suspicious_urls.append(url)

    pair_counter = Counter()
    per_worker_ip_counter = defaultdict(Counter)
    for ip, d in per_ip_count.items():
        for worker, cnt in d.items():
            pair_counter[worker, ip] = cnt
            per_worker_ip_counter[worker][ip] = cnt

    ip_to_worker = {}
    worker_to_ip = {}

    manual_list = [
        ('47.15.0.101', 'A1PZ6FQ0WT3ROZ'),
        ("103.66.214.8", "A16OJTSFLK977U"),
        ("98.181.162.144", "A2QKAA5YS0P4CI"),
        ("172.58.139.199", "A2ZLJQWCM8KU36"),
        ('157.46.125.118', 'A3I7ZBU31VKOMC'),
        ('45.130.83.147', 'A2PD9SHVWNX7Q2'),
        ('42.106.179.175', 'A1VBAI9GBDQSMO'),
        ('157.49.214.154', 'A19Q990ORTCBN4'),
        ('177.189.186.207', 'A1FOPTNEQ0XL18'),
        ('158.51.119.46', 'A249LDVPG27XCE'),
        ('72.69.8.225', 'A19LVWX8ZLO6CS'),
    ]
    for ip, worker in manual_list:
        ip_to_worker[ip] = worker
        worker_to_ip[worker] = ip

    for worker, cnt in worker_hit_count.most_common():
        if cnt > 0 and worker not in worker_to_ip:
            d = {}
            for ip, ip_cnt in per_worker_ip_counter[worker].most_common():
                if ip not in ip_to_worker:
                    d[ip] = ip_cnt, ip_worker_time_match[ip, worker]

            print(worker, cnt, d)
    for ip, cnt in ip_count.most_common():
        if ip not in ip_to_worker and cnt:
            d = {}
            for worker, w_cnt in per_ip_count[ip].items():
                if worker not in worker_to_ip and w_cnt:
                    d[worker] = w_cnt
            print(ip, cnt, d)
        else:
            pass

    for (worker, ip), cnt in pair_counter.most_common():
        if worker not in worker_to_ip and ip not in ip_to_worker:
            if cnt > 0:
                ip_to_worker[ip] = worker
                worker_to_ip[worker] = ip
                print("{}/{}={}/{}/{}".format(ip, worker, ip_count[ip], worker_hit_count[worker], pair_counter[worker, ip]))

    print(ip_to_worker)

    print(len(raw_worker_count))
    print("num workers", len(worker_hit_count))
    print("num ips", len(per_ip_count))

    for url in suspicious_urls:
        hits = url_grouped_hits[url]
        logs = url_grouped_logs[url]
        print(url)
        for hit in hits:
            ip = worker_to_ip[hit.worker_id] if hit.worker_id in worker_to_ip else "?"
            print(hit.worker_id, ip, ip in ips)
        ips = set([log.ip for log in logs])
        for ip in ips:
            print(ip)


    suspicious_hit = []
    return suspicious_hit


def test_apache_time():
    s = "04/Jul/2021:16:24:11 -0400"
    time = datetime.datetime.strptime(s, "%d/%b/%Y:%H:%M:%S %z")
    EDT = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(EDT)
    print(EDT)
    print('parsed_time', time)
    print(now)
    print(now - time)


def test_mturk_time():
    s = 'Sat Jul 03 23:28:53 PDT 2021'
    t = parse_mturk_time(s)
    print(t)
    print(t.tzinfo)


if __name__ == "__main__":
    test_mturk_time()
