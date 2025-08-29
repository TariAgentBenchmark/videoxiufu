#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2025.08
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @Blog   :  https://viatorsun.blog.csdn.net
# @Note   :


import json

import httpx

TOKEN_URL = "http://10.114.100.172:5580/api/token/get_cached_token"
REFRESHED_URL = "http://10.114.100.172:5580/api/token/get_refreshed_token"


async def get_new_url(video_url, token_url=TOKEN_URL, refreshed_url=REFRESHED_URL):
    token = await get_token(token_url)
    verified_video_url = f"{video_url}?tk={token}"

    token = await get_token(refreshed_url)
    return verified_video_url, token


async def get_token(token_url=TOKEN_URL):
    # 从地址获取TOKEN
    async with httpx.AsyncClient() as client:
        resp = await client.get(token_url)
        data = resp.json()
        if isinstance(data, str):
            data = json.loads(data)
        try:
            token = data["token"]
            return token
        except:
            print(
                f"token接口获取内容为{data}, 数据类型为{type(data)}，没法用token = data['token']"
            )
            return 0


# 获取/生成 task_id
def obtain_taskID(data: json):
    cor_id = data["cor_id"]
    video_id = data["video_id"]

    task_id = video_id + "_" + cor_id
    return task_id


POST_ASSESSMENT = "http://10.164.147.226/pvms-jm-api/api/quality/fix_ret"
GET_ASSESSMENT = "http://10.164.147.226/pvms-jm-api/api/quality/fix_stopped"


async def send_fix_ret(result, token):
    """
    Example result:
    {
        "video_id": "video_id",
        "cor_id": "cor_id",
        "segment_id": segment_id,
        "before_segment_url": before_segment_url,
        "after_segment_url": after_segment_url,
        "before_frame": before_frame_b64,
        "after_frame" : after_frame_b64
    }
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "MyHttpxClient/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # use async httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(POST_ASSESSMENT, json=result, headers=headers)
        response.raise_for_status()
        return response.json()


async def send_fix_stopped(result, token):
    """
    Example result:
    {
        "video_id": "video_id",
        "cor_id": "cor_id",
        "video_url": "video_url",
        "description": "视频已结束"
    }
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "MyHttpxClient/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(GET_ASSESSMENT, params=result, headers=headers)
        response.raise_for_status()
        return response.json()