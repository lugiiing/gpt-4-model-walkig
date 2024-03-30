import io
import gradio as gr
import cv2
import base64
import openai
import os
import asyncio
import concurrent.futures
from openai import AsyncOpenAI

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from PIL import Image
import ast
import matplotlib.pyplot as plt


from prompts import VISION_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, FINAL_EVALUATION_SYSTEM_PROMPT, FINAL_EVALUATION_USER_PROMPT, SUMMARY_AND_TABLE_PROMPT
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

global global_dict
global_dict = {}

######
# SETTINGS
VIDEO_FRAME_LIMIT = 2000

######

def validate_api_key():
    #client = openai.api_key

    try:
        # Make your OpenAI API request here
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Hello world"},
            ]
        )
    except openai.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        response = None
        error = e
        pass
    except openai.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        response = None
        error = e
        pass
    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        response = None
        error = e
        pass

    if response:
        return True
    else:
        raise gr.Error(f"OpenAI returned an API Error: {error}")


def _process_video(video_file):
    # Read and process the video file
    video = cv2.VideoCapture(video_file.name)

    if 'video_file' not in global_dict:
        global_dict.setdefault('video_file', video_file.name)
    else:
        global_dict['video_file'] = video_file.name

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    if len(base64Frames) > VIDEO_FRAME_LIMIT:
        raise gr.Warning(f"Video's play time is too long. (>1m)")
    print(len(base64Frames), "frames read.")

    if not base64Frames:
        raise gr.Error(f"Cannot open the video.")
    return base64Frames


def _make_video_batch(video_file):

    frames = _process_video(video_file)

    TOTAL_FRAME_COUNT = len(frames)
    BATCH_SIZE = int(1)
    TOTAL_BATCH_SIZE = int(TOTAL_FRAME_COUNT * 5 / 100)  # 5 = total_batch_percent
    BATCH_STEP = int(TOTAL_FRAME_COUNT / TOTAL_BATCH_SIZE)
    
    base64FramesBatch = []

    for idx in range(0, TOTAL_FRAME_COUNT, BATCH_STEP * BATCH_SIZE):
        #print(f'## {idx}')
        temp = []
        for i in range(BATCH_SIZE):
            #print(f'# {idx + BATCH_STEP * i}')
            if (idx + BATCH_STEP * i) < TOTAL_FRAME_COUNT:
                temp.append(frames[idx + BATCH_STEP * i])
            else:
                continue
        base64FramesBatch.append(temp)
    
    for idx, batch in enumerate(base64FramesBatch):
        # assert len(batch) <= BATCH_SIZE
        print(f'##{idx} - batch_size: {len(batch)}')

    if 'batched_frames' not in global_dict:
        global_dict.setdefault('batched_frames', base64FramesBatch)
    else:
        global_dict['batched_frames'] = base64FramesBatch

    return base64FramesBatch


def show_batches(video_file):
    
    batched_frames = _make_video_batch(video_file) 
    
    images1 = []
    for i, l in enumerate(batched_frames):
        print(f"#### Batch_{i+1}")
        for j, img in enumerate(l):
            print(f'## Image_{j+1}')
            image_bytes = base64.b64decode(img.encode("utf-8"))
            # Convert the bytes to a stream (file-like object)
            image_stream = io.BytesIO(image_bytes)
            # Open the image as a PIL image
            image = Image.open(image_stream)
            images1.append((image, f"batch {i+1}"))
        print("-"*100)
    
    return images1



# 각 버튼에 대한 액션 함수 정의

rubrics = ""
rubrics_keyword = ''


def action_weight_shift():
    rubric_a = "1. Fouse on Lower body-Feet-Weight Shift, Step on your toes, centered on the front third of the ball of your foot and perpendicular to the floor. 2. Focus on Lower body-Feet-Weight Shift, Follow the line on the floor, stepping out in a figure of 1 (closer to a figure of 11 for men). 3. Focus on Lower body-Knee-Weight Shift,Walk so that both knees are touching."
    rubric_b = "4. Focus on Upper body- Shoulder Weight Shift, Be careful not to let your shoulders slump too far back and shift your center of gravity backwards. 5. Focus on Upper body Weight Shift, Keep your upper body straight and move your legs along vertically without losing balance."
    rubric_subsets = {'a':rubric_a, 'b':rubric_b}
    rubrics_keyword = '"Feet Weight Shift", "Feet Walking Line", "Knee Weight Shift", "Shoulder Weight Shift", "Upper Body Weight Shift"'
    global_dict['rubric_subsets'] = rubric_subsets
    global_dict['rubrics_keyword'] = rubrics_keyword
    return rubric_subsets, rubrics_keyword

def action_balance():
    rubric_a = "1. Focus on Lower body-Leg Balance, Walk so that your hips or pelvis do not sway violently when you bring your legs together in a straight line. 2. Focus on Upper body-Arm Balance,your left and right arms should have the same angle and shape so that they are balanced."
    rubric_b = "3. Focus on Upper body- Head Balance, Keep your gaze and face straight ahead at all times, and do not let your head dip or tilt to the side. 4. Focus on Upper body- Shoulder Balance, your shoulders should not be tilted to one side, and your shoulders and ears should be in the following positions. 5. Focus on Upper body-Posture Balance, Your pelvis, back, shoulders, and neck should be symmetrical from side to side."
    rubric_subsets = {'a':rubric_a, 'b':rubric_b}
    rubrics_keyword = '"Leg Balance", "Arm Balance", "Gaze", "Shoulder Balance", "Posture Balance"'
    global_dict['rubric_subsets'] = rubric_subsets
    global_dict['rubrics_keyword'] = rubrics_keyword
    return rubric_subsets, rubrics_keyword

def action_form():
    rubric_a = "1. Focus on Upper body-Arm Position, Walk with your arms at the side of your pelvis/hips, with a natural extension and not excessive bending. 2. Focus on Upper body-Arm Form, The angle of the bent elbow should be about 45 degrees forward and 15 degrees backward, with the arms extending more forward. "
    rubric_b = "3. Focus on Upper body-Hand Form,Your hands should be slightly bent and made into a fist, as if you're holding a small ball or egg at your fingertips. 4. Focus on Upper body-Hand Form and Hand Position, Make a fist with your thumbs facing forward and move your arms across your thighs."
    rubric_c = "5. Focus on Upper body-Head form, Your chin should be drawn in towards your chest about 5 degrees with your neck stretched out, not overly lifted. 6. Focus on Upper body- Shoulder Form,Open your shoulders and lower back straight and without tension and then round them back about 3 degrees. "
    rubric_d = "7. Focus on Upper body- Chest Form, Keep your chest open, bring your ribs together in the back on both sides. 8. Focus on Lower body-Feet Form, Don't walk with your toes together or in a limp position where your toes point outward. "
    rubric_subsets = {'a':rubric_a, 'b':rubric_b, 'c':rubric_c, 'd':rubric_d}
    rubrics_keyword = '"Arm Position", "Arm Angle", "Hand Form", "Hand Position", "Chin", "Shoulder Angle", "Chest", "Leg Form"'
    global_dict['rubric_subsets'] = rubric_subsets
    global_dict['rubrics_keyword'] = rubrics_keyword
    return rubric_subsets, rubrics_keyword

def action_all():
    rubric_a = "1. Fouse on Lower body-Feet-Weight Shift, Step on your toes, centered on the front third of the ball of your foot and perpendicular to the floor. 2. Focus on Lower body-Feet-Weight Shift, Follow the line on the floor, stepping out in a figure of 1 (closer to a figure of 11 for men). 3. Focus on Lower body-Knee-Weight Shift,Walk so that both knees are touching. 4. Focus on Upper body- Shoulder Weight Shift, Be careful not to let your shoulders slump too far back and shift your center of gravity backwards. 5. Focus on Upper body Weight Shift, Keep your upper body straight and move your legs along vertically without losing balance."
    rubric_b = "6. Focus on Lower body-Leg Balance, Walk so that your hips or pelvis do not sway violently when you bring your legs together in a straight line. 7. Focus on Upper body-Arm Balance,Your left and right arms should have the same angle and shape so that they are balanced. 8. Focus on Upper body- Head Balance, Keep your gaze and face straight ahead at all times, and don't let your head dip or tilt to the side. 9. Focus on Upper body- Shoulder Balance,Your shoulders should not be tilted to one side, and your shoulders and ears should be in the following positions. 10. Focus on Upper body-Posture Balance, Your pelvis, back, shoulders, and neck should be symmetrical from side to side."
    rubric_c = "11. Focus on Upper body-Arm Position, Walk with your arms at the side of your pelvis/hips, with a natural extension and not excessive bending. 12. Focus on Upper body-Arm Form, The angle of the bent elbow should be about 45 degrees forward and 15 degrees backward, with the arms extending more forward. 13. Focus on Upper body-Hand Form,Your hands should be slightly bent and made into a fist, as if you're holding a small ball or egg at your fingertips. 14. Focus on Upper body-Hand Form and Hand Position, Make a fist with your thumbs facing forward and move your arms across your thighs."
    rubric_d = "15. Focus on Upper body-Head form, Your chin should be drawn in towards your chest about 5 degrees with your neck stretched out, not overly lifted. 16. Focus on Upper body- Shoulder Form,Open your shoulders and lower back straight and without tension and then round them back about 3 degrees. 17. Focus on Upper body- Chest Form, Keep your chest open, bring your ribs together in the back on both sides. 18. Focus on Lower body-Feet Form, Do not walk with your toes together or in a limp position where your toes point outward. "
    rubric_subsets = {'a':rubric_a, 'b':rubric_b, 'c':rubric_c, 'd':rubric_d}
    rubrics_keyword = '"Feet Weight Shift", "Feet Walking Line", "Knee Weight Shift", "Shoulder Weight Shift", "Upper Body Weight Shift", "Leg Balance", "Arm Balance", "Gaze", "Shoulder Balance", "Posture Balance", "Arm Position", "Arm Angle", "Hand Form", "Hand Position", "Chin", "Shoulder Angle", "Chest", "Leg Form"'
    global_dict['rubric_subsets'] = rubric_subsets
    global_dict['rubrics_keyword'] = rubrics_keyword
    return rubric_subsets, rubrics_keyword


async def async_call_gpt_vision(client, batch, rubric_subset):
    # Format the messages for the vision prompt, including the rubric subset and images in the batch
    vision_prompt_messages = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},  # Ensure VISION_SYSTEM_PROMPT is defined
        {
            "role": "user",
            "content": [
                PromptTemplate.from_template(USER_PROMPT_TEMPLATE).format(rubrics=rubric_subset),  # Ensure USER_PROMPT_TEMPLATE is defined
                *map(lambda x: {"image": x, "resize": 300}, batch),
            ],
        },
    ]
    
    # Parameters for the API call
    params = {
        "model": "gpt-4-vision-preview",
        "messages": vision_prompt_messages,
        "max_tokens": 1024,
    }

    # Asynchronous API call
    try:
        result_raw = await client.chat.completions.create(**params)
        result = result_raw.choices[0].message.content
        print(result)
        return result
    except Exception as e:
        print(f"Error processing batch with rubric subset {rubric_subset}: {e}")
        return None
    

async def process_rubrics_in_batches(client, frames, rubric_subsets):
    
    results = {}
    for key, rubric_subset in rubric_subsets.items():
        # Process each image batch with the current rubric subset
        tasks = [async_call_gpt_vision(client, batch, rubric_subset) for batch in frames]
        subset_results = await asyncio.gather(*tasks)
        results[key] = [result for result in subset_results if result is not None]

    # Filter out None results in case of errors
    return results

def wrapper_call_gpt_vision():
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    frames = global_dict.get('batched_frames')
    rubric_subsets = global_dict.get('rubric_subsets')

    async def call_gpt_vision():
        async_full_result_vision = await process_rubrics_in_batches(client, frames, rubric_subsets)
        if 'full_result_vision' not in global_dict:
            global_dict.setdefault('full_result_vision', async_full_result_vision)
        else:
            global_dict['full_result_vision'] = async_full_result_vision
        return async_full_result_vision
    
    # 새 이벤트 루프 생성 및 설정
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(call_gpt_vision())


async def async_get_evaluation_text(client, result_subset):
    
    result_subset_text = ' \n'.join(result_subset)
    evaluation_text = PromptTemplate.from_template(FINAL_EVALUATION_USER_PROMPT).format(evals = result_subset_text)

    evaluation_text_message = [
        {"role": "system", "content": FINAL_EVALUATION_SYSTEM_PROMPT},  # Ensure VISION_SYSTEM_PROMPT is defined
        {
            "role": "user",
            "content": evaluation_text,
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": evaluation_text_message,
        "max_tokens": 1024,
    }

    # Asynchronous API call
    try:
        result_raw_2 = await client.chat.completions.create(**params)
        result_2 = result_raw_2.choices[0].message.content
        return result_2
    except Exception as e:
        print(f"Error getting evaluation text {result_subset}: {e}")
        return None

#    return evaluation_text

async def async_get_full_result(client, full_result_vision):
    
    #tasks = []
    results_2 = {}
    # Create a task for each entry in full_result_vision and add to tasks list
    for key, result_subset in full_result_vision.items():
        tasks_2 = [async_get_evaluation_text(client, result_subset)]
        text_results = await asyncio.gather(*tasks_2)
        results_2[key] = [result_2 for result_2 in text_results if result_2 is not None]
    

    results_2_val_list = list(results_2.values())
    results_2_val = ""
    for i in range(len(results_2_val_list)):
        results_2_val += results_2_val_list[i][0]
        results_2_val += "\n"

    return results_2_val
    # Combine all results into a single string


def wrapper_get_full_result():
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    full_result_vision = global_dict.get('full_result_vision')

    #{key: choice.choices[0].message.content for key, choice in full_result_vision.items()}

    async def get_full_result():
        full_text = await async_get_full_result(client,full_result_vision)        
        # global_dict에 결과를 올바르게 저장
        if 'full_text' not in global_dict:
            global_dict.setdefault('full_text', full_text)
        else:
            global_dict['full_text'] = full_text  # 새 값으로 초기화
        print("full_text: ")
        print(full_text)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(get_full_result())


def get_final_anser():
    rubrics_keyword = global_dict.get('rubrics_keyword')
    full_text = global_dict.get('full_text')


    chain = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4",
        max_tokens=1024,
        temperature=0,
    )
    prompt = PromptTemplate.from_template(SUMMARY_AND_TABLE_PROMPT)
    
    runnable = prompt | chain | StrOutputParser()
    final_eval = runnable.invoke({"full_text": full_text, "rubrics_keyword":rubrics_keyword})

    print(final_eval)
    
    if 'final_eval' not in global_dict:
        global_dict.setdefault('final_eval', final_eval)
    else:
        global_dict['final_eval'] = final_eval
    
    return final_eval


def tablize_final_anser():

    final_eval = global_dict.get('final_eval')
    pos3 = int(final_eval.find("**table**"))
    pos4 = int(final_eval.find("]]"))
    tablize_final_eval = ast.literal_eval(final_eval[(pos3+10):(pos4+2)])


    cat_final_eval, val_final_eval = tablize_final_eval[0], tablize_final_eval[1]
    val_final_eval = [int(score) for score in val_final_eval]
    
    
    fig, ax = plt.subplots()
    ax.bar(cat_final_eval, val_final_eval)
    ax.set_ylabel('Scores')
    ax.set_title('Scores by category')
    #plt.xticks(rotation=30)
    plt.rc('xtick', labelsize=3)
    ax.set_xticks(range(len(cat_final_eval)))
    ax.set_yticks([0,2,4,6,8,10])

    ax.set_xticklabels(cat_final_eval)

    # PIL.Image 객체로 변환
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)  
    buf.seek(0)  

    # PIL.Image 객체로 변환
    image = Image.open(buf)
    return image


def breif_final_anser():
    final_eval = global_dict.get('final_eval')
    pos1 = int(final_eval.find("**Total score**"))
    pos2 = int(final_eval.find("----END of the summary----"))
    breif_final_eval = final_eval[pos1:pos2]
    return breif_final_eval

def fin_final_anser():
    full_text = global_dict.get('full_text')
    fin_final_eval = full_text
    return fin_final_eval


def mainpage():
    with gr.Blocks() as start_page:
        gr.Markdown("Title")
        with gr.Row():
            with gr.Column(scale=1):
                start_button = gr.Button("start")
    
        gr.Markdown("비디오 업로드 페이지")
        with gr.Row():
            with gr.Column(scale=1):
                video_upload = gr.File(
                    label="Upload your video (video under 1 minute is the best..!)",
                    file_types=["video"],
                )

            with gr.Column(scale=1):
                weight_shift_button = gr.Button("Weight Shift")
                balance_button = gr.Button("Balance")
                form_button = gr.Button("Form")
                overall_button = gr.Button("Overall")

        with gr.Row():
            with gr.Column(scale=1):
                process_button = gr.Button("Process")

        gr.Markdown("결과 페이지")
        with gr.Row():
            with gr.Column(scale=1):

                output_box_fin_table = gr.Image(type="pil", label="Score Chart")

            with gr.Column(scale=1):
                output_box_fin_brief = gr.Textbox(
                    label="Brief Evaluation",
                    lines=10,
                    interactive=True,
                    show_copy_button=True,
                )

        with gr.Row():
            with gr.Column(scale=1):

                output_box_fin_fin = gr.Textbox(
                    label="Detailed Evaluation",
                    lines=10,
                    interactive=True,
                    show_copy_button=True,
                )
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="Batched Snapshots of Video",
                    columns=[3],
                    rows=[10],
                    object_fit="contain",
                    height="auto",
                )


        #start_button.click(fn = video_rubric, inputs=[], outputs= [])
        weight_shift_button.click(fn = action_weight_shift, inputs=[], outputs=[])
        balance_button.click(fn = action_balance, inputs=[], outputs=[])
        form_button.click(fn = action_form, inputs=[], outputs=[])
        overall_button.click(fn = action_all, inputs=[], outputs=[])
        process_button.click(fn=show_batches, inputs=[video_upload], outputs=[gallery])\
            .success(fn=lambda: wrapper_call_gpt_vision(), inputs=[], outputs=[]) \
            .success(fn=lambda: wrapper_get_full_result(), inputs=[], outputs=[])\
            .success(fn=get_final_anser, inputs=[], outputs=[])\
            .success(fn=tablize_final_anser, inputs=[], outputs=[output_box_fin_table])\
            .success(fn=breif_final_anser, inputs=[], outputs=[output_box_fin_brief])\
            .success(fn=fin_final_anser, inputs=[], outputs=[output_box_fin_fin])  #output=None? []?

    start_page.launch()



if __name__ == "__main__":
    mainpage()
