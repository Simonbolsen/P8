import asyncio
import time

from status_setter import discord_status, send_discord_message

from os.path import exists
path_to_token = "./src/discord/token"

token = None
if exists(path_to_token):
    with open(path_to_token) as f:
        token = f.read()



status_setter = discord_status()
async def run_discord_bot():
    print("Test1")
    await status_setter.start(token, 1095627677848834128)
    print("Test2")
    await status_setter.set_started()
    print("Test3")


# asyncio.create_task(run_discord_bot())

# loop = asyncio.get_event_loop()
# tasks = [
#     loop.create_task(run_discord_bot())
# ]
# loop.run_forever(asyncio.wait(tasks))
# loop.close()


send_discord_message(token, 1095627677848834128, "Test")


while True:
    time.sleep(5)
    print("sleep")