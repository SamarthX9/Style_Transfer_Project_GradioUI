import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder("vgg_normalised.pth").to(device)
decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        "experiment/final_exp/decoder_final.pth",
        map_location=device
    )
)

encoder.eval()
decoder.eval()

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def tensor_to_pil(image):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)

    return transforms.ToPILImage()(image)


def generate(content_image, style_image, alpha):
    if content_image is None or style_image is None:
        raise gr.Error("Please upload both a content image and a style image.")

    output = style_transfer(
        content_image,
        style_image,
        encoder,
        decoder,
        alpha,
        device
    )

    return tensor_to_pil(output)


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="AdaIN Neural Style Transfer"
) as demo:

    gr.Markdown("""
    # 🎨 AdaIN Neural Style Transfer

    Transform any content image into an artwork using
    **Adaptive Instance Normalization (AdaIN)**.

    Upload a content image and a style image,
    adjust the style strength, and generate your stylized result.
    """)

    with gr.Row():
        content = gr.Image(
            type="pil",
            label="📷 Content Image",
            height=350
        )

        style = gr.Image(
            type="pil",
            label="🎨 Style Image",
            height=350
        )

    alpha = gr.Slider(
        minimum=0,
        maximum=1,
        value=1,
        step=0.1,
        label="🎚 Style Strength"
    )

    generate_btn = gr.Button(
        "✨ Generate Stylized Image",
        variant="primary"
    )

    output = gr.Image(
        type="pil",
        label="🖼 Stylized Output",
        height=500
    )

    generate_btn.click(
        fn=generate,
        inputs=[content, style, alpha],
        outputs=output
    )



    gr.Markdown("""
    ---
    Built with **PyTorch**, **Gradio**, and **Adaptive Instance Normalization (AdaIN)**.
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)))





