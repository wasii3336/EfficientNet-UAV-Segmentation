
  {
   "cell_type": "code",
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/gdrive')\n",
    "%cd /gdrive"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "UDYGB0kvRKML",
    "outputId": "412bdfe2-cdfd-4e95-de99-6b39e458a49b"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [],
   "metadata": {
    "id": "ccJEapEQRNAK"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "!pip install segmentation-models-pytorch albumentations -q\n",
    "import torch\n",
    "print(\"GPU available:\", torch.cuda.is_available())\n",
    "print(\"GPU name:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "vJ6BdRj-RYj8",
    "outputId": "1fdd22e9-0e07-4b7c-826c-a9253d846bdb"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "dataset_path = \"/content/drive/MyDrive/VDD_project/VDD\"\n",
    "for split in [\"train\", \"val\", \"test\"]:\n",
    "    src = os.path.join(dataset_path, split, \"src\")\n",
    "    gt  = os.path.join(dataset_path, split, \"gt\")\n",
    "    print(f\"{split}: {len(os.listdir(src))} images, {len(os.listdir(gt))} masks\")"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 176
    },
    "id": "0S6NyP_xRhyx",
    "outputId": "bd045208-44b2-445a-e7b7-e08d97916018"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "\n",
    "# Search for VDD folder in your Drive\n",
    "base = \"/content/drive/MyDrive\"\n",
    "for item in os.listdir(base):\n",
    "    print(item)"
   ],
   "metadata": {
    "id": "eOcIOVXORq-Z"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "\n",
    "# Search for VDD folder in your Drive\n",
    "base = \"/content/drive/MyDrive\"\n",
    "for item in os.listdir(base):\n",
    "    print(item)"
   ],
   "metadata": {
    "id": "iVw8q3IqRa5w"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [],
   "metadata": {
    "id": "hUf8DHfBRy1x"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n"
   ],
   "metadata": {
    "id": "ASM2kFvvR2cF"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "\n",
    "base = \"/content/drive/MyDrive\"\n",
    "for item in os.listdir(base):\n",
    "    print(item)"
   ],
   "metadata": {
    "id": "hEcqkZr3R4xA"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "\n",
    "vdd_path = \"/content/drive/MyDrive/VDD_project\"\n",
    "for item in os.listdir(vdd_path):\n",
    "    print(item)"
   ],
   "metadata": {
    "id": "sxjRrCJFSBpy"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "\n",
    "dataset_path = \"/content/drive/MyDrive/VDD_project/VDD\"\n",
    "for split in [\"train\", \"val\", \"test\"]:\n",
    "    src = os.path.join(dataset_path, split, \"src\")\n",
    "    gt  = os.path.join(dataset_path, split, \"gt\")\n",
    "    print(f\"{split}: {len(os.listdir(src))} images, {len(os.listdir(gt))} masks\")"
   ],
   "metadata": {
    "id": "6ybqTE0-SI8p"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "\n",
    "# Check what's inside VDD_project\n",
    "print(\"Inside VDD_project:\")\n",
    "for item in os.listdir(\"/content/drive/MyDrive/VDD_project\"):\n",
    "    print(\" \", item)"
   ],
   "metadata": {
    "id": "hIX9a5VYSTXa"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "# Check what's inside VDD_project/VDD\n",
    "print(\"Inside VDD_project/VDD:\")\n",
    "for item in os.listdir(\"/content/drive/MyDrive/VDD_project/VDD\"):\n",
    "    print(\" \", item)"
   ],
   "metadata": {
    "id": "q6ERaLBBSaJr"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "\n",
    "print(\"Inside VDD_project/VDD/VD:\")\n",
    "for item in os.listdir(\"/content/drive/MyDrive/VDD_project/VDD/VD\"):\n",
    "    print(\" \", item)"
   ],
   "metadata": {
    "id": "IIJHKJ1eSeKl"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "\n",
    "print(\"Inside VDD_project/VDD/VDD:\")\n",
    "for item in os.listdir(\"/content/drive/MyDrive/VDD_project/VDD/VD\"):\n",
    "    print(\" \", item)"
   ],
   "metadata": {
    "id": "HoyXCNyFSjCp"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "\n",
    "print(\"Inside VDD_project/VDD/VDD:\")\n",
    "for item in os.listdir(\"/content/drive/MyDrive/VDD_project/VDD/VDD\"):\n",
    "    print(\" \", item)"
   ],
   "metadata": {
    "id": "v8vR4zTpSnca"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "\n",
    "dataset_path = \"/content/drive/MyDrive/VDD_project/VDD/VDD\"\n",
    "for split in [\"train\", \"val\", \"test\"]:\n",
    "    src = os.path.join(dataset_path, split, \"src\")\n",
    "    gt  = os.path.join(dataset_path, split, \"gt\")\n",
    "    print(f\"{split}: {len(os.listdir(src))} images, {len(os.listdir(gt))} masks\")"
   ],
   "metadata": {
    "id": "h1HuHCrASriJ"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os, sys, yaml, numpy as np, torch, torch.nn as nn, cv2, glob\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from PIL import Image\n",
    "from tqdm import tqdm\n",
    "import segmentation_models_pytorch as smp\n",
    "\n",
    "# \u2500\u2500 Config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "DATASET_PATH = \"/content/drive/MyDrive/VDD_project/VDD/VDD\"\n",
    "IMAGE_SIZE   = 512\n",
    "BATCH_SIZE   = 4\n",
    "EPOCHS       = 20\n",
    "LR           = 0.0001\n",
    "NUM_CLASSES  = 7\n",
    "CHECKPOINT   = \"/content/drive/MyDrive/VDD_project/efficientnet_clean_best.pth\"\n",
    "\n",
    "# \u2500\u2500 Corruptions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "def add_fog(image, severity=2):\n",
    "    fog_intensity = [0.3, 0.5, 0.7, 0.85, 0.95][severity - 1]\n",
    "    fog_layer = np.ones_like(image, dtype=np.float32) * 255\n",
    "    result = image.astype(np.float32) * (1 - fog_intensity) + fog_layer * fog_intensity\n",
    "    return np.clip(result, 0, 255).astype(np.uint8)\n",
    "\n",
    "def add_gaussian_noise(image, severity=2):\n",
    "    std = [0.04, 0.06, 0.09, 0.17, 0.26][severity - 1] * 255\n",
    "    noise = np.random.normal(0, std, image.shape).astype(np.float32)\n",
    "    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)\n",
    "\n",
    "def add_blur(image, severity=2):\n",
    "    k = [3, 5, 7, 9, 11][severity - 1]\n",
    "    return cv2.GaussianBlur(image, (k, k), 0)\n",
    "\n",
    "def reduce_brightness(image, severity=2):\n",
    "    factor = [0.8, 0.6, 0.45, 0.3, 0.15][severity - 1]\n",
    "    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)\n",
    "\n",
    "def reduce_contrast(image, severity=2):\n",
    "    factor = [0.8, 0.6, 0.4, 0.2, 0.1][severity - 1]\n",
    "    mean = np.mean(image)\n",
    "    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)\n",
    "\n",
    "CORRUPTIONS = {\n",
    "    \"fog\": add_fog,\n",
    "    \"gaussian_noise\": add_gaussian_noise,\n",
    "    \"blur\": add_blur,\n",
    "    \"brightness\": reduce_brightness,\n",
    "    \"contrast\": reduce_contrast,\n",
    "}\n",
    "\n",
    "# \u2500\u2500 Dataset \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "class VDDDataset(Dataset):\n",
    "    def __init__(self, dataset_path, split=\"train\", image_size=512, augmentation=False, corruption=None, severity=2):\n",
    "        self.image_size  = image_size\n",
    "        self.augmentation = augmentation\n",
    "        self.corruption  = corruption\n",
    "        self.severity    = severity\n",
    "        self.image_dir   = os.path.join(dataset_path, split, \"src\")\n",
    "        self.mask_dir    = os.path.join(dataset_path, split, \"gt\")\n",
    "        self.images      = sorted(glob.glob(os.path.join(self.image_dir, \"*.JPG\")))\n",
    "        print(f\"[{split}] Found {len(self.images)} images\")\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.images)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        img_path  = self.images[idx]\n",
    "        name      = os.path.splitext(os.path.basename(img_path))[0]\n",
    "        mask_path = os.path.join(self.mask_dir, name + \".png\")\n",
    "        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)\n",
    "        mask  = np.array(Image.open(mask_path))\n",
    "        if self.augmentation:\n",
    "            corruption = np.random.choice(list(CORRUPTIONS.keys()))\n",
    "            image = CORRUPTIONS[corruption](image, severity=np.random.randint(1, 4))\n",
    "        if self.corruption:\n",
    "            image = CORRUPTIONS[self.corruption](image, self.severity)\n",
    "        image = cv2.resize(image, (self.image_size, self.image_size))\n",
    "        mask  = cv2.resize(mask,  (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)\n",
    "        mask  = np.clip(mask, 0, 6)\n",
    "        image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)\n",
    "        mask  = torch.from_numpy(mask).long()\n",
    "        return image, mask\n",
    "\n",
    "# \u2500\u2500 Metric \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "def compute_miou(preds, labels, num_classes=7):\n",
    "    preds, labels = preds.cpu().numpy(), labels.cpu().numpy()\n",
    "    ious = []\n",
    "    for cls in range(num_classes):\n",
    "        inter = ((preds == cls) & (labels == cls)).sum()\n",
    "        union = ((preds == cls) | (labels == cls)).sum()\n",
    "        if union > 0:\n",
    "            ious.append(inter / union)\n",
    "    return np.mean(ious) if ious else 0.0\n",
    "\n",
    "# \u2500\u2500 Model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Using device: {device}\")\n",
    "\n",
    "model = smp.DeepLabV3Plus(\n",
    "    encoder_name    = \"efficientnet-b0\",\n",
    "    encoder_weights = \"imagenet\",\n",
    "    in_channels     = 3,\n",
    "    classes         = NUM_CLASSES,\n",
    ").to(device)\n",
    "\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=LR)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "\n",
    "# \u2500\u2500 DataLoaders \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "train_dl = DataLoader(VDDDataset(DATASET_PATH, \"train\", IMAGE_SIZE, augmentation=False),\n",
    "                      batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)\n",
    "val_dl   = DataLoader(VDDDataset(DATASET_PATH, \"val\",   IMAGE_SIZE, augmentation=False),\n",
    "                      batch_size=BATCH_SIZE, shuffle=False, num_workers=2)\n",
    "\n",
    "# \u2500\u2500 Training Loop \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "best_miou = 0.0\n",
    "for epoch in range(1, EPOCHS + 1):\n",
    "    model.train()\n",
    "    total_loss = 0\n",
    "    for images, masks in tqdm(train_dl, desc=f\"Epoch {epoch}/{EPOCHS} [Train]\"):\n",
    "        images, masks = images.to(device), masks.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        loss = criterion(model(images), masks)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        total_loss += loss.item()\n",
    "\n",
    "    model.eval()\n",
    "    miou_scores = []\n",
    "    with torch.no_grad():\n",
    "        for images, masks in tqdm(val_dl, desc=f\"Epoch {epoch}/{EPOCHS} [Val]\"):\n",
    "            images, masks = images.to(device), masks.to(device)\n",
    "            preds = torch.argmax(model(images), dim=1)\n",
    "            miou_scores.append(compute_miou(preds, masks))\n",
    "\n",
    "    avg_miou = np.mean(miou_scores)\n",
    "    avg_loss = total_loss / len(train_dl)\n",
    "    print(f\"Epoch {epoch} | Loss: {avg_loss:.4f} | Val mIoU: {avg_miou:.4f}\")\n",
    "\n",
    "    if avg_miou > best_miou:\n",
    "        best_miou = avg_miou\n",
    "        torch.save(model.state_dict(), CHECKPOINT)\n",
    "        print(f\"  Saved best model! mIoU: {best_miou:.4f}\")\n",
    "\n",
    "print(f\"\\nTraining complete! Best mIoU: {best_miou:.4f}\")"
   ],
   "metadata": {
    "id": "HuJWnKVTSzww"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/gdrive')"
   ],
   "metadata": {
    "id": "uxBqmTFLTDv0"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "# Change these two lines:\n",
    "DATASET_PATH = \"/gdrive/MyDrive/VDD_project/VDD/VDD\"\n",
    "CHECKPOINT   = \"/gdrive/MyDrive/VDD_project/efficientnet_clean_best.pth\""
   ],
   "metadata": {
    "id": "2YWy0sDdUHzp"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/gdrive')\n",
    "\n",
    "import os, sys, yaml, numpy as np, torch, torch.nn as nn, cv2, glob\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from PIL import Image\n",
    "from tqdm import tqdm\n",
    "import segmentation_models_pytorch as smp\n",
    "\n",
    "# \u2500\u2500 Config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "DATASET_PATH = \"/gdrive/MyDrive/VDD_project/VDD/VDD\"\n",
    "IMAGE_SIZE   = 512\n",
    "BATCH_SIZE   = 4\n",
    "EPOCHS       = 20\n",
    "LR           = 0.0001\n",
    "NUM_CLASSES  = 7\n",
    "CHECKPOINT   = \"/gdrive/MyDrive/VDD_project/efficientnet_clean_best.pth\"\n",
    "\n",
    "# \u2500\u2500 Corruptions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "def add_fog(image, severity=2):\n",
    "    fog_intensity = [0.3, 0.5, 0.7, 0.85, 0.95][severity - 1]\n",
    "    fog_layer = np.ones_like(image, dtype=np.float32) * 255\n",
    "    result = image.astype(np.float32) * (1 - fog_intensity) + fog_layer * fog_intensity\n",
    "    return np.clip(result, 0, 255).astype(np.uint8)\n",
    "\n",
    "def add_gaussian_noise(image, severity=2):\n",
    "    std = [0.04, 0.06, 0.09, 0.17, 0.26][severity - 1] * 255\n",
    "    noise = np.random.normal(0, std, image.shape).astype(np.float32)\n",
    "    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)\n",
    "\n",
    "def add_blur(image, severity=2):\n",
    "    k = [3, 5, 7, 9, 11][severity - 1]\n",
    "    return cv2.GaussianBlur(image, (k, k), 0)\n",
    "\n",
    "def reduce_brightness(image, severity=2):\n",
    "    factor = [0.8, 0.6, 0.45, 0.3, 0.15][severity - 1]\n",
    "    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)\n",
    "\n",
    "def reduce_contrast(image, severity=2):\n",
    "    factor = [0.8, 0.6, 0.4, 0.2, 0.1][severity - 1]\n",
    "    mean = np.mean(image)\n",
    "    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)\n",
    "\n",
    "CORRUPTIONS = {\n",
    "    \"fog\": add_fog,\n",
    "    \"gaussian_noise\": add_gaussian_noise,\n",
    "    \"blur\": add_blur,\n",
    "    \"brightness\": reduce_brightness,\n",
    "    \"contrast\": reduce_contrast,\n",
    "}\n",
    "\n",
    "# \u2500\u2500 Dataset \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "class VDDDataset(Dataset):\n",
    "    def __init__(self, dataset_path, split=\"train\", image_size=512, augmentation=False, corruption=None, severity=2):\n",
    "        self.image_size   = image_size\n",
    "        self.augmentation = augmentation\n",
    "        self.corruption   = corruption\n",
    "        self.severity     = severity\n",
    "        self.image_dir    = os.path.join(dataset_path, split, \"src\")\n",
    "        self.mask_dir     = os.path.join(dataset_path, split, \"gt\")\n",
    "        self.images       = sorted(\n",
    "            glob.glob(os.path.join(self.image_dir, \"*.JPG\")) +\n",
    "            glob.glob(os.path.join(self.image_dir, \"*.jpg\"))\n",
    "        )\n",
    "        print(f\"[{split}] Found {len(self.images)} images\")\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.images)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        img_path  = self.images[idx]\n",
    "        name      = os.path.splitext(os.path.basename(img_path))[0]\n",
    "        mask_path = os.path.join(self.mask_dir, name + \".png\")\n",
    "        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)\n",
    "        mask  = np.array(Image.open(mask_path))\n",
    "        if self.augmentation:\n",
    "            corruption = np.random.choice(list(CORRUPTIONS.keys()))\n",
    "            image = CORRUPTIONS[corruption](image, severity=np.random.randint(1, 4))\n",
    "        if self.corruption:\n",
    "            image = CORRUPTIONS[self.corruption](image, self.severity)\n",
    "        image = cv2.resize(image, (self.image_size, self.image_size))\n",
    "        mask  = cv2.resize(mask,  (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)\n",
    "        mask  = np.clip(mask, 0, 6)\n",
    "        image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)\n",
    "        mask  = torch.from_numpy(mask).long()\n",
    "        return image, mask\n",
    "\n",
    "# \u2500\u2500 Metric \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "def compute_miou(preds, labels, num_classes=7):\n",
    "    preds, labels = preds.cpu().numpy(), labels.cpu().numpy()\n",
    "    ious = []\n",
    "    for cls in range(num_classes):\n",
    "        inter = ((preds == cls) & (labels == cls)).sum()\n",
    "        union = ((preds == cls) | (labels == cls)).sum()\n",
    "        if union > 0:\n",
    "            ious.append(inter / union)\n",
    "    return np.mean(ious) if ious else 0.0\n",
    "\n",
    "# \u2500\u2500 Model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Using device: {device}\")\n",
    "\n",
    "model = smp.DeepLabV3Plus(\n",
    "    encoder_name    = \"efficientnet-b0\",\n",
    "    encoder_weights = \"imagenet\",\n",
    "    in_channels     = 3,\n",
    "    classes         = NUM_CLASSES,\n",
    ").to(device)\n",
    "\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=LR)\n",
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "\n",
    "# \u2500\u2500 DataLoaders \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "train_dl = DataLoader(VDDDataset(DATASET_PATH, \"train\", IMAGE_SIZE, augmentation=True),\n",
    "                      batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)\n",
    "val_dl   = DataLoader(VDDDataset(DATASET_PATH, \"val\",   IMAGE_SIZE, augmentation=False),\n",
    "                      batch_size=BATCH_SIZE, shuffle=False, num_workers=2)\n",
    "\n",
    "# \u2500\u2500 Training Loop \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "best_miou = 0.0\n",
    "for epoch in range(1, EPOCHS + 1):\n",
    "    model.train()\n",
    "    total_loss = 0\n",
    "    for images, masks in tqdm(train_dl, desc=f\"Epoch {epoch}/{EPOCHS} [Train]\"):\n",
    "        images, masks = images.to(device), masks.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        loss = criterion(model(images), masks)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        total_loss += loss.item()\n",
    "\n",
    "    model.eval()\n",
    "    miou_scores = []\n",
    "    with torch.no_grad():\n",
    "        for images, masks in tqdm(val_dl, desc=f\"Epoch {epoch}/{EPOCHS} [Val]\"):\n",
    "            images, masks = images.to(device), masks.to(device)\n",
    "            preds = torch.argmax(model(images), dim=1)\n",
    "            miou_scores.append(compute_miou(preds, masks))\n",
    "\n",
    "    avg_miou = np.mean(miou_scores)\n",
    "    avg_loss = total_loss / len(train_dl)\n",
    "    print(f\"Epoch {epoch} | Loss: {avg_loss:.4f} | Val mIoU: {avg_miou:.4f}\")\n",
    "    scheduler.step()\n",
    "\n",
    "    if avg_miou > best_miou:\n",
    "        best_miou = avg_miou\n",
    "        torch.save(model.state_dict(), CHECKPOINT)\n",
    "        print(f\"  Saved best model! mIoU: {best_miou:.4f}\")\n",
    "\n",
    "print(f\"\\nTraining complete! Best mIoU: {best_miou:.4f}\")"
   ],
   "metadata": {
    "id": "NbOjdt8FUY60",
    "outputId": "c16a97c8-83bf-4d51-d317-d3d763f5a0b6",
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000,
     "referenced_widgets": [
      "c39c34e6b3744a779e7331a8753aae13",
      "28411b86edae462589fa3a6b737da630",
      "1bcfb378fad94ee5be987146f803ea11",
      "b5980ddd6b6c49829611f2e4711c2dea",
      "84ff82b3c7cb42959bfc8af9fe999d40",
      "de7fb04b90d342e5a0343ee298389004",
      "69e907d5a454400cbce25fe346cc7951",
      "444988eedc0f47788e11e9c74d926aac",
      "99f906aa88cf4fb8927e5d5a76c1b5ae",
      "844e32998cca42d1bd2288e8d102da74",
      "1c2b702ba59647ce9325ee8f0a63cc88",
      "0d8c7a8b75884b65a4c3a621df230a91",
      "4b20eec09a504efcbff51b74bdbcae72",
      "c2f51efa3fcb4d6ca2c85071e0cbfd74",
      "7b589f40f373495581c2670f203d4016",
      "bd4b6900641641889f12e32ee81acb7f",
      "740ee1e3d9524b5f8f5c2de33663f80e",
      "81cb9af42e274faebae745c737abb043",
      "bfda1c04914d4283b1826c3fc1803e39",
      "39a16a8b0f454dd58fb2e56d4e24d135",
      "3a00b6140f264eb7b44d8dc56137f59a",
      "5bdb49d2a6fa428b9ce5c92734cd66f8"
     ]
    }
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "!find /content/drive/MyDrive -name \"*.ipynb\" 2>/dev/null"
   ],
   "metadata": {
    "id": "uOrDufMapPVu"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive', force_remount=True)\n",
    "\n",
    "!find /content/drive/MyDrive -name \"*.ipynb\" 2>/dev/null"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "047Xb37WpbRe",
    "outputId": "74c5154c-7f63-4837-e93a-2fb08bae7911"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "sizes = [\n",
    "    '/content/drive/MyDrive/VDD_project/EfficientNet_UAV_Segmentation.ipynb',\n",
    "    '/content/drive/MyDrive/Colab Notebooks/EfficientNet_UAV_Segmentation.ipynb',\n",
    "    '/content/drive/MyDrive/Colab Notebooks/Copy of EfficientNet_UAV_Segmentation.ipynb'\n",
    "]\n",
    "for f in sizes:\n",
    "    print(os.path.getsize(f), f)"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "qruONW1Opjm2",
    "outputId": "08069097-813a-4b98-f6d2-8cfa99cc2777"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os, shutil\n",
    "os.chdir('/content/drive/MyDrive/VDD_project')\n",
    "\n",
    "# Rename .bak back to .ipynb\n",
    "shutil.copy(\n",
    "    '/content/drive/MyDrive/VDD_project/EfficientNet_UAV_Segmentation.ipynb.bak',\n",
    "    '/content/drive/MyDrive/VDD_project/EfficientNet_UAV_Segmentation.ipynb'\n",
    ")\n",
    "\n",
    "!git add EfficientNet_UAV_Segmentation.ipynb\n",
    "!git status"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "vXrsPwabrDQZ",
    "outputId": "d2709030-66f4-462a-f159-b58a14932965"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "files = os.listdir('/content/drive/MyDrive/VDD_project')\n",
    "print(files)"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "9xzEtV4erXxw",
    "outputId": "8e65338d-a84c-46c3-d627-92cc3fc468ab"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import os\n",
    "os.chdir('/content/drive/MyDrive/VDD_project')\n",
    "!git status\n",
    "!git log --oneline -3"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Y_L1lz9irzxe",
    "outputId": "3539619e-9570-4b59-bba1-5fdb20f6a8fb"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import json, os\n",
    "\n",
    "path = '/content/drive/MyDrive/VDD_project/EfficientNet_UAV_Segmentation.ipynb'\n",
    "\n",
    "with open(path, 'r') as f:\n",
    "    nb = json.load(f)\n",
    "\n",
    "# Fix the metadata.widgets issue\n",
    "if 'widgets' in nb.get('metadata', {}):\n",
    "    del nb['metadata']['widgets']\n",
    "\n",
    "with open(path, 'w') as f:\n",
    "    json.dump(nb, f)\n",
    "\n",
    "print(\"Fixed!\")"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "dgXQb8MqsBj_",
    "outputId": "8dd4325a-68e4-46c9-ddd1-2b9eb3eb54a1"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [
    "import json\n",
    "\n",
    "path1 = '/content/drive/MyDrive/VDD_project/EfficientNet_UAV_Segmentation.ipynb'\n",
    "path2 = '/content/drive/MyDrive/Colab Notebooks/EfficientNet_UAV_Segmentation.ipynb'\n",
    "path3 = '/content/drive/MyDrive/Colab Notebooks/Copy of EfficientNet_UAV_Segmentation.ipynb'\n",
    "\n",
    "for p in [path1, path2, path3]:\n",
    "    with open(p, 'r') as f:\n",
    "        nb = json.load(f)\n",
    "    cells = nb.get('cells', [])\n",
    "    # Print first code cell content\n",
    "    for c in cells:\n",
    "        if c['cell_type'] == 'code' and c['source']:\n",
    "            print(f\"\\n--- {p} ---\")\n",
    "            print(''.join(c['source'][:200]))\n",
    "            break"
   ],
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "n76P7u25sftV",
    "outputId": "2ad93b80-8bdf-43ea-fb34-18b199e1bb71"
   },
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "source": [],
   "metadata": {
    "id": "IuBi0HOjxdzc"
   },
   "execution_count": null,
   "outputs": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": [],
   "gpuType": "T4"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "accelerator": "GPU"
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
