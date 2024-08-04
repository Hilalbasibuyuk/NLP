from diffusers import StableDiffusionPipeline

model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

model = model.to("cuda")

prompt = "A fantasy landscape with mountains and a river"
image = model(prompt).images[0]

image.save("fantasy_landscape.png")
