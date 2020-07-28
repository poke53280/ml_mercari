

from time import sleep
from json import dumps, loads
from kafka import KafkaProducer
from kafka import KafkaConsumer


producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda x: dumps(x).encode('utf-8'))


for e in range(10):
    data = {'number' : e}
    producer.send('apples', value=data)
    sleep(5)





