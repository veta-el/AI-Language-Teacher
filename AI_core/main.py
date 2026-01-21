import asyncio
import websockets
import json

from config_llm import Config_LLM

import response_llm

import time

async def handle_websocket(websocket):
    async def check_timeout(): #Func for trigger first message
        while True:
            global start_time
            if time.time() - start_time >= 900: #Every 15 min
                response, rus = await response_llm.choose_first_msg ()
                print(f'Server-side output First Message: {response}')
                start_time = time.time()
                await websocket.send(json.dumps({'message':response}))
                await websocket.send(json.dumps({'rus':rus}))
            await asyncio.sleep(2)

    print('Client connected')
    global start_time
    start_time = time.time()
    async_task = asyncio.create_task(check_timeout())

    try:  
        while True:
            is_audio = False
            message = await websocket.recv() #Receive msg

            user_input = json.loads(message)['user_input']
            params = json.loads(message)['params']
            start_time = time.time() #Reset trigger for first msg

            print(f'Server-side params: {params}')

            if json.loads(message)['audio'] == 'true': #If we get signal about audio, wait for audio input
                print(f'Server-side input: audio')
                is_audio = True
                user_input = await websocket.recv()
            else:
                print(f'Server-side input: {user_input}')
                if user_input == Config_LLM.STOPPER: #Stopper logic
                    await websocket.send (json.dumps({'message':Config_LLM.FINAL_PHRASE}))
                    break

            response, rus, add_answer = await response_llm.process_input (user_input, params, is_audio) #Process input and get response, translation, additional answer if needed

            if type(response) != str:
                if params ['ch_audio_output']: #Send audio as output
                    print(f'Server-side output: {"audio"}')
                    await websocket.send(response)
                    await websocket.send(json.dumps({'rus':rus}))
            else: #Send text as output
                print(f'Server-side output: {response}')
                await websocket.send(json.dumps({'message':response}))
                await websocket.send(json.dumps({'rus':rus}))
            
            if params ['ch_gen_answers'] and add_answer:
                print(f'Server-side output - generated answers')
                await websocket.send(json.dumps({'add_answer':add_answer}))

    except websockets.ConnectionClosed:
        print('Client disconnected')
        pass  
    async_task.cancel()


if __name__ == '__main__': #Main func
    async def main():
        async with websockets.serve(handle_websocket, 'localhost', 3000) as ws:
            print ('Main server connected')
            await ws.wait_closed()

    asyncio.run(main())
