import asyncio
import discord
import typing
import threading
from os.path import exists


class discord_bot(discord.Client):
    keep_alive:bool = True
    # async def _on_ready(self):
    #     print('Logged on as', self.user)
    #     await self.on_connected_callback()

        # while self.keep_alive:
        #     await asyncio.sleep(10)

    async def on_message(self, message):
        # don't respond to ourselves
        if message.author == self.user:
            return

        if message.content == 'ping':
            await message.channel.send('pong')

def get_discord_bot(token:str) -> typing.Awaitable[discord.Client]:
    future = asyncio.Future()

    intents = discord.Intents.default()
    bot = discord_bot(intents=intents)
    async def on_ready():
        future.set_result(bot)
        # while bot.keep_alive:
        #     await asyncio.sleep(10)
    
    bot.on_ready = on_ready

    # bot.on_error = lambda ex: future.set_exception(ex)

    class bot_thread(threading.Thread):
        def __init__(self, thread_name):
            threading.Thread.__init__(self)
            self.thread_name = thread_name

        def run(self):
            bot.run(token)

    async def run_bot():
        bot.run(token)

    thread = bot_thread("Discord bot")
    thread.start()

    # asyncio.threads.to_thread(runner())
    
    # async def runner():
    #     async with bot:
    #         await bot.start(token)
    # task = asyncio.create_task(runner())
    # task.set_name("discord bot")

    # threading.Thread(bot.start(token)).start()
    
    return future

class discord_status:
    bot:discord_bot
    channel_id:int

    def __init__(self) -> None:
        pass

    async def start(self, token:str, channel_id:int):
        self.channel_id = channel_id
        self.bot = await get_discord_bot(token)

    async def close(self):
        self.keep_alive = False
        await self.bot.close()


    async def set_started(self):
        await self.bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="Ray eat my RAM and burn my CPU"))
        # await self.channel.send("Started")

    async def set_done(self):
        await self.bot.change_presence(activity=None)
        # await self.channel.send("Done")


def send_discord_message(token_path:str, channel_id:int, message:str) -> typing.Awaitable:
    token = None
    if exists(token_path):
        with open(token_path) as f:
            token = f.read()
    else:
        return None

    future = asyncio.Future()

    intents = discord.Intents.default()
    bot = discord.Client(intents=intents)
    async def on_ready():
        channel = bot.get_channel(channel_id)
        await channel.send(message)
        await bot.close()
        future.set_result(None)
    
    bot.on_ready = on_ready
    bot.on_error = lambda: future.set_exception()
    asyncio.run(bot.start(token))

    return future