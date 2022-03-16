# DL4media-captioning

## Image Captioning

Returns latvian captions to any provided images. The release conatins a pre-trained model ready to caption.

**To run** download everything from the latest release, then start the docker with "docker compose up --build". The server accepts images and then returns captions for the passed image in latvian.

---

Model was trained on a combination of the COCO dataset with the captions translated to latvian, and a similar amount of picture-caption pairs taken from the LETA news agency database.

Model based on TensorFlow Authors work on Image Captioning with Attention.

Model is trained with a vocabulary of 23,000 words; words outside of these top 23,000 were replaced with the token <unk>, and these can sometimes be seen in the output captions.
