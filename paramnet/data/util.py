from pathlib import Path
from typing import Any, Dict, List, Optional

import rosbag


class BagData:
    def __init__(self, bag_path: Path, topics: Optional[List[str]] = None) -> None:
        bag = rosbag.Bag(bag_path)
        topics = (
            topics
            if topics is not None
            else self.bag.get_type_and_topic_info()[1].keys()
        )

        self.data: Dict[str, List[Any]] = {}
        for topic in topics:
            self.data[topic] = []
        for topic, msg, _ in self.bag.read_messages(topics=topics):
            self.data[topic].append(msg)

        bag.close()

    def __getitem__(self, topic: str) -> list:
        return self.data[topic]
