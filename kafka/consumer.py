

from time import sleep
from json import dumps, loads
from kafka import KafkaProducer
from kafka import KafkaConsumer


consumer = KafkaConsumer(
    'apples',
     bootstrap_servers=['localhost:9092'],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id='my-group')

for message in consumer:
    message = message.value
    print(message)



