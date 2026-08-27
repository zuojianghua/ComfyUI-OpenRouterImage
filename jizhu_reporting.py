"""图片节点复用机杼共享客户端上报单张最终成品。"""

from io import BytesIO

from jizhu_comfy_client import JizhuComfyClient, QuotaExhaustedError


def _report_failed(client, execution, model, provider):
    """尽力记录模型调用失败，避免上报故障遮蔽原始错误。"""
    try:
        client.report_failed(
            execution,
            media_type="image",
            requested_units=1,
            model=model,
            provider=provider,
        )
    except Exception:
        pass


def report_image(model, provider, generate):
    """先校验额度，再执行一次模型调用并上报其唯一结果。"""
    try:
        client = JizhuComfyClient()
        execution = client.start_execution()
        client.check_quota("image", 1, model)
    except QuotaExhaustedError as exc:
        return None, f"额度不足：{exc}"
    except Exception as exc:
        return None, f"机杼额度校验失败：{exc}"

    try:
        image, status = generate()
    except Exception as exc:
        _report_failed(client, execution, model, provider)
        return None, str(exc)
    if image is None:
        _report_failed(client, execution, model, provider)
        return None, status

    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        client.report_completed(
            execution,
            media_type="image",
            requested_units=1,
            model=model,
            provider=provider,
            filename=f"{execution.execution_id}.png",
            content_type="image/png",
            binary=buffer.getvalue(),
        )
    except Exception as exc:
        return None, f"机杼结果上报失败：{exc}"
    return image, status
