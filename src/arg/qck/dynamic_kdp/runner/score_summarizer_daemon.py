from arg.qck.dynamic_kdp.score_summarizer import ScoreSummarizer

if __name__ == "__main__":
    worker = ScoreSummarizer()
    worker.file_watch_daemon()