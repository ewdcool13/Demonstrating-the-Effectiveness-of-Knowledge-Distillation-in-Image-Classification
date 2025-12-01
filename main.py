# ==============================
# 0. CIFAR-100 다운로드 및 라이브러리/환경 준비
# ==============================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import numpy as np
import os
import time
from functools import wraps
from torchvision.models import mobilenet_v2

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")

# 시드 고정
seed = 42
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(seed)
np.random.seed(seed)

# CIFAR-100 데이터셋 다운로드 및 DataLoader 준비
batch_size = 64
test_batch_size = 100

# 데이터 전처리
transform_train = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# 학습/테스트 데이터셋
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

# 학습용 DataLoader (shuffle=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
# 테스트용 DataLoader
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2
)

print(f"학습 데이터 개수: {len(train_dataset)}, 테스트 데이터 개수: {len(test_dataset)}")

# ★ [FIX 1] Soft label 생성을 위한 고정 순서 DataLoader (shuffle=False)
train_loader_for_soft = torch.utils.data.DataLoader(
    datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train),
    batch_size=batch_size, shuffle=False, num_workers=2
)

# ==============================
# 1. 실행 시간 측정용 데코레이터
# ==============================

import time
from functools import wraps

def logging_time(func):
    """
    함수 실행 시간을 측정하고 출력하는 데코레이터
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[{func.__name__}] 실행 시간: {elapsed:.2f}초")
        return result
    return wrapper


# ==============================
# 2. ResNet 모델 정의
# ==============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------- Basic Block 정의 ----------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --------- ResNet 클래스 정의 ----------
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ==============================
# 3. MobileNetV2 (CIFAR-100, 32x32 입력 전용)
# ==============================

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetV2_CIFAR100(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNetV2_CIFAR100, self).__init__()
        # ImageNet 구조 기반으로 불러오되, 첫 Conv 수정
        self.model = mobilenet_v2(pretrained=False)

        # 첫 Conv2d를 CIFAR용으로 조정 (kernel 3, stride 1, padding 1)
        self.model.features[0][0] = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 마지막 분류기 수정
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ==============================
# 4. ResNet50 원본 학습 및 Soft Label 생성
# ==============================

@logging_time
def train_resnet50_for_soft_labels(train_loader, test_loader, device, num_epochs=20, lr=0.1):
    """
    ResNet50 원본 모델 학습 ([3,4,6,3] 블록 구성) - Soft label 생성용

    사용 예시:
        resnet50_model = train_resnet50_for_soft_labels(train_loader, test_loader, device, num_epochs=20)
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Test Acc: {acc*100:.2f}%")

    print(f"ResNet50 학습 완료 - Best Acc: {best_acc*100:.2f}%")
    return model
2
# ResNet50 원본 학습 (Soft label 생성용)
print("\n=== ResNet50 원본 학습 시작 (Soft label 생성용) ===")
resnet50_model = train_resnet50_for_soft_labels(train_loader, test_loader, device, num_epochs=20)



# ==============================
# 4-2. ResNet50 Soft Label 생성
# ==============================

@logging_time
def generate_soft_labels(model, data_loader, device):
    """
    학습된 모델을 사용해 soft label 생성
    :return: soft_labels [N, C], targets [N]
    """
    model.eval()
    soft_labels = []
    targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)              # logits
            probs = F.softmax(output, dim=1)  # 확률 분포 (soft label)
            soft_labels.append(probs.cpu())
            targets.append(target)

    soft_labels = torch.cat(soft_labels, dim=0)
    targets = torch.cat(targets, dim=0)
    print(f"Soft label 생성 완료: {soft_labels.shape}")
    return soft_labels, targets


# ★ [FIX 2] soft label 생성 시 shuffle=False인 loader 사용
soft_labels, hard_targets = generate_soft_labels(
    resnet50_model, train_loader_for_soft, device
)

# 필요하면 저장
torch.save({'soft_labels': soft_labels, 'targets': hard_targets},
           'resnet50_soft_labels.pt')
print("Soft label 데이터 저장 완료")

# ==============================
# 5. Soft → Hard 변환 (α 값 적용)
# ==============================

import torch

@logging_time
def soft_to_hard_labels(soft_labels, alphas):
    """
    Soft label을 alpha 값(= Hard label 비율)에 따라 샘플 단위로 hard/soft를 혼합
    - 각 샘플의 최대 확률(max probability)을 기준으로 정렬
    - 상위 alpha 비율의 샘플만 hard label(one-hot)로 변환
    - 나머지 (1-alpha) 비율의 샘플은 원래 soft label을 그대로 유지

    :param soft_labels: [num_samples, num_classes] tensor
    :param alphas: list of alpha 값 (0~1) - Hard label로 변환할 샘플 비율
    :return: dict {alpha: transformed_labels}

    사용 예시:
        alphas = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        alpha_labels = soft_to_hard_labels(soft_labels, alphas)
    """
    num_samples, num_classes = soft_labels.shape
    alpha_labels = {}

    max_probs, max_indices = soft_labels.max(dim=1)
    sorted_probs, sorted_indices = torch.sort(max_probs, descending=True)

    for alpha in alphas:
        transformed = soft_labels.clone()

        if alpha <= 0.0:
            alpha_labels[alpha] = transformed
            print(f"α={alpha} 변환 완료 (모든 샘플 Soft 유지)")
            continue

        if alpha >= 1.0:
            transformed.zero_()
            transformed.scatter_(1, max_indices.unsqueeze(1), 1.0)
            alpha_labels[alpha] = transformed
            print(f"α={alpha} 변환 완료 (모든 샘플 Hard)")
            continue

        num_hard = max(1, int(round(alpha * num_samples)))
        hard_indices = sorted_indices[:num_hard]

        transformed[hard_indices] = 0.0
        transformed[hard_indices, max_indices[hard_indices]] = 1.0

        alpha_labels[alpha] = transformed
        print(f"α={alpha} 변환 완료 (Hard 샘플 수: {num_hard}/{num_samples})")

    return alpha_labels

# --------- 사용 예시 ----------
alphas = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
alpha_labels = soft_to_hard_labels(soft_labels, alphas)

# 저장 예시
torch.save(alpha_labels, "alpha_labels.pt")
print("α 변환 label 저장 완료")

# Custom Dataset 클래스 (soft/hard label 사용용)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, idx):
        data, _ = self.dataset[idx]
        label = self.labels[idx]
        return data, label

    def __len__(self):
        return len(self.dataset)



# ==============================
# 6. test_model() : Student 학습 + 성능 측정
# ==============================

def test_model(model, hard_ratio=1.0,
               train_loader=None, test_loader=None,
               device=None, epochs=[5, 10, 15, 20]):
    """
    모델을 학습하고 지정된 epoch에서의 정확도와 학습 시간을 측정하는 함수

    hard_ratio:
      - 1.0 : GT hard label (일반 학습)
      - 0.0~1.0 : teacher soft label 기반 (α = hard_ratio 비율만 one-hot 변환)
    """
    import time as _time

    # Hard ratio 검증
    if not (0.0 <= hard_ratio <= 1.0):
        raise ValueError("hard_ratio는 0.0~1.0 사이의 값이어야 합니다.")

    # 전역 변수 사용 (기본값이 None인 경우)
    if train_loader is None:
        train_loader = globals().get('train_loader')
    if test_loader is None:
        test_loader = globals().get('test_loader')
    if device is None:
        device = globals().get('device')

    if train_loader is None or test_loader is None or device is None:
        raise ValueError("train_loader, test_loader, device가 정의되지 않았습니다.")

    # Label 설정 (hard_ratio에 따라)
    if hard_ratio < 1.0:
        alpha = float(hard_ratio)
        alpha = round(alpha, 2)  # 키 일치
        alpha_labels_dict = globals().get('alpha_labels')
        if alpha_labels_dict is None or alpha not in alpha_labels_dict:
            raise ValueError(
                f"alpha={alpha}에 해당하는 alpha_labels가 없습니다. "
                "먼저 soft_to_hard_labels를 실행하세요."
            )
        custom_labels = alpha_labels_dict[alpha]

        # Custom Dataset 및 DataLoader 생성
        custom_dataset = CustomDataset(globals().get('train_dataset'), custom_labels)
        custom_train_loader = torch.utils.data.DataLoader(
            custom_dataset, batch_size=globals().get('batch_size', 64),
            shuffle=True, num_workers=2
        )
        train_loader = custom_train_loader
        label_info = f"Teacher soft/hard label (hard 비율: {hard_ratio*100:.0f}%, alpha={alpha})"
    else:
        # Hard label (원래 GT 사용)
        label_info = "Ground-truth hard label"

    # 모델을 디바이스로 이동
    model = model.to(device)

    # 모델 이름 확인
    if isinstance(model, ResNet):
        model_name = "ResNet"
    elif isinstance(model, MobileNetV2_CIFAR100):
        model_name = "MobileNetV2"
    else:
        model_name = type(model).__name__

    print(f"\n{'='*70}")
    print(f"{model_name} 모델 학습 시작 ({label_info})")
    print(f"{'='*70}")

    # Optimizer 설정
    if isinstance(model, ResNet):
        optimizer = optim.SGD(model.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=1e-4)
    else:  # MobileNet 등
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    results = {}
    max_epoch = max(epochs)

    total_start_time = _time.time()
    print(f"\n총 {max_epoch} epoch까지 학습합니다. ({epochs} epoch에서 정확도 및 시간 측정)\n")

    for epoch in range(1, max_epoch + 1):
        epoch_start_time = _time.time()

        # Training
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            # ★ [FIX 3] soft label일 때는 항상 분포 기준 KLDiv로 학습
            if hard_ratio < 1.0:
                # target: [B, C] 확률 분포 (soft + 일부 one-hot)
                loss = F.kl_div(
                    F.log_softmax(output, dim=1),
                    target,
                    reduction='batchmean'
                )
            else:
                # GT hard label (정수 클래스)
                loss = F.cross_entropy(output, target)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        acc = correct / total
        epoch_time = _time.time() - epoch_start_time
        total_time = _time.time() - total_start_time

        if epoch in epochs:
            results[epoch] = {
                "accuracy": acc,
                "time": total_time,
                "epoch_time": epoch_time
            }
            print(f"Epoch [{epoch}/{max_epoch}] - Loss: {avg_loss:.4f}, "
                  f"Test Acc: {acc*100:.2f}%, "
                  f"총 학습 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        elif epoch % 5 == 0:
            print(f"Epoch [{epoch}/{max_epoch}] - Loss: {avg_loss:.4f}, "
                  f"Test Acc: {acc*100:.2f}%")

    total_time = _time.time() - total_start_time

    print(f"\n{'='*70}")
    print(f"{model_name} 학습 완료 - 결과 요약")
    print(f"{'='*70}")
    print(f"{'Epoch':<10} {'정확도 (%)':<15} {'총 학습 시간 (초)':<20} {'총 학습 시간 (분)':<15}")
    print("-"*70)

    for epoch in epochs:
        if epoch in results:
            acc = results[epoch]["accuracy"] * 100
            time_sec = results[epoch]["time"]
            time_min = time_sec / 60
            print(f"{epoch:<10} {acc:<15.2f} {time_sec:<20.2f} {time_min:<15.2f}")

    print(f"{'='*70}\n")

    return results


# ==============================
# 사용 예시 (주석 처리 - 필요시 주석 해제하여 실행)
# ==============================
#
# # 1. Hard label로 ResNet 학습
# resnet_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)
# results_hard = test_model(resnet_model, 1.0, epochs=[5, 10, 15, 20])
#
# # 2. 20% Hard label로 ResNet 학습
# results_20hard = test_model(resnet_model, 0.2, epochs=[5, 10, 15, 20])
#
# # 3. 완전 Soft label로 ResNet 학습
# results_soft = test_model(resnet_model, 0.0, epochs=[5, 10, 15, 20])
#
# # 4. 다른 블록 구성으로 테스트
# resnet_50 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=100)
# results_50 = test_model(resnet_50, 0.2, epochs=[5, 10, 15, 20])

# # === MobileNetV2 모델 테스트 ===
# # 5. Hard label로 MobileNetV2 학습
# mobilenet_model = MobileNetV2_CIFAR100(num_classes=100)
# mobilenet_hard_results = test_model(mobilenet_model, 1.0, epochs=[5, 10, 15, 20])
#
# # 6. 80% Hard label로 MobileNetV2 학습
# mobilenet_model = MobileNetV2_CIFAR100(num_classes=100)
# mobilenet_20hard_results = test_model(mobilenet_model, 0.8, epochs=[5, 10, 15, 20])
#
# # 7. 60% Hard label로 MobileNetV2 학습
# mobilenet_model = MobileNetV2_CIFAR100(num_classes=100)
# mobilenet_20hard_results = test_model(mobilenet_model, 0.6, epochs=[5, 10, 15, 20])
#
# # 8. 40% Hard label로 MobileNetV2 학습
# mobilenet_model = MobileNetV2_CIFAR100(num_classes=100)
# mobilenet_20hard_results = test_model(mobilenet_model, 0.4, epochs=[5, 10, 15, 20])
#
# # 9. 20% Hard label로 MobileNetV2 학습
# mobilenet_model = MobileNetV2_CIFAR100(num_classes=100)
# mobilenet_20hard_results = test_model(mobilenet_model, 0.2, epochs=[5, 10, 15, 20])
#
# # 10. 완전 Soft label로 MobileNetV2 학습
# mobilenet_model = MobileNetV2_CIFAR100(num_classes=100)
# mobilenet_soft_results = test_model(mobilenet_model, 0.0, epochs=[5, 10, 15, 20])
