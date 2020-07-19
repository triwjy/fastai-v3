import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=19yW2561TVqjaD7cO7AMBay-F5Lth2fid'
export_file_name = 'stage3_deploy.pkl'

classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    label_dict =   {'00': 'long-dress',
                    '01': 'dress',
                    '02': 'shirt',
                    '03': 'sweater',
                    '04': 'jeans',
                    '05': 'ring',
                    '06': 'earring',
                    '07': 'hat',
                    '08': 'clutch',
                    '09': 'carry-bag',
                    '10': 'cellphone-cover',
                    '11': 'cellphone',
                    '12': 'clock',
                    '13': 'feeding-bottle',
                    '14': 'rice cooker',
                    '15': 'coofee powder',
                    '16': 'women shoes',
                    '17': 'high-heels',
                    '18': 'air-conditioner/remote',
                    '19': 'flashdrive',
                    '20': 'chair',
                    '21': 'racket',
                    '22': 'biking helmet',
                    '23': 'gloves',
                    '24': 'watch',
                    '25': 'belt',
                    '26': 'airphone',
                    '27': 'vehicle toys',
                    '28': 'jacket',
                    '29': 'trousers',
                    '30': 'sneakers',
                    '31': 'snacks',
                    '32': 'mask',
                    '33': 'desinfectant',
                    '34': 'skin-care product',
                    '35': 'perfume',
                    '36': 'home utilities',
                    '37': 'laptop',
                    '38': 'eating container',
                    '39': 'vase',
                    '40': 'shower/bidet head',
                    '41': 'sofa'}
    result = label_dict[prediction]
    return JSONResponse({'result': str(result)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
