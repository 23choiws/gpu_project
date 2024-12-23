import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from timm import create_model  # Timm 라이브러리 사용
from torchvision.datasets import ImageNet
#import matplotlib.pyplot as plt
import random
import torch.quantization
import copy
import os
import time
from torch.nn.quantized.dynamic import Linear as DynamicQuantizedLinear

import quantize_extension 

torch.cuda.empty_cache()

USE_CUDA = torch.cuda.is_available()  # GPU를 사용할 수 있는지 확인
device = torch.device("cuda" if USE_CUDA else "cpu")  # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)
#device = "cpu"
random.seed(777)
torch.manual_seed(777)  
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


learning_rate = 0.0005
training_epochs = 20
batch_size = 128
drop_prob = 0.3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
mnist_test = dsets.MNIST(root='MNIST_data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

class MLP_model(nn.Module):
    def __init__(self, hidden_size=5500, drop_prob=0.3, num_layers=31):
        super(MLP_model, self).__init__()

        # self.linear1 = nn.Linear(784, 512, bias=True)
        # self.batchnorm1 = nn.BatchNorm1d(512)
        # #self.relu1 = nn.ReLU()
        # self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        # self.dropout1 = nn.Dropout(p=drop_prob)
        
        # self.linear2 = nn.Linear(512, 512, bias=True)
        # self.batchnorm2 = nn.BatchNorm1d(512)
        # #self.relu2 = nn.ReLU()
        # self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        # self.dropout2 = nn.Dropout(p=drop_prob)
        
        # self.linear3 = nn.Linear(512, 512, bias=True)
        # self.batchnorm3 = nn.BatchNorm1d(512)
        # #self.relu3 = nn.ReLU()
        # self.relu3 = nn.LeakyReLU(negative_slope=0.01)
        # self.dropout3 = nn.Dropout(p=drop_prob)
        
        # self.linear4 = nn.Linear(512, 10, bias=True)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear1 = nn.Linear(784, hidden_size, bias=True)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # 나머지 숨겨진 레이어들
        for i in range(2, num_layers + 1):
            setattr(self, f'linear{i}', nn.Linear(hidden_size, hidden_size, bias=True))
            setattr(self, f'batchnorm{i}', nn.BatchNorm1d(hidden_size))
            setattr(self, f'relu{i}', nn.LeakyReLU(negative_slope=0.01))
            setattr(self, f'dropout{i}', nn.Dropout(p=drop_prob))
        
        # self.linear2 = nn.Linear(10000, 10000, bias=True)
        # self.batchnorm2 = nn.BatchNorm1d(10000)
        # self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        # self.dropout2 = nn.Dropout(p=drop_prob)
        
        # self.linear3 = nn.Linear(10000, 10000, bias=True)
        # self.batchnorm3 = nn.BatchNorm1d(10000)
        # self.relu3 = nn.LeakyReLU(negative_slope=0.01)
        # self.dropout3 = nn.Dropout(p=drop_prob)
        
        self.linear_out = nn.Linear(hidden_size, 10, bias=True)
        
        # Xavier uniform 초기화
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # nn.init.xavier_uniform_(self.linear2.weight)
        # nn.init.xavier_uniform_(self.linear3.weight)
        # nn.init.xavier_uniform_(self.linear4.weight)
        
        # 숨겨진 레이어 초기화
        for i in range(2, self.num_layers + 1):
            linear = getattr(self, f'linear{i}')
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        # 편향 초기화
        # nn.init.zeros_(self.linear1.bias)
        # nn.init.zeros_(self.linear2.bias)
        # nn.init.zeros_(self.linear3.bias)

        # nn.init.xavier_uniform_(self.linear4.weight)
        # nn.init.zeros_(self.linear4.bias)

        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # 나머지 숨겨진 레이어들 통과
        for i in range(2, self.num_layers + 1):
            linear = getattr(self, f'linear{i}')
            batchnorm = getattr(self, f'batchnorm{i}')
            relu = getattr(self, f'relu{i}')
            dropout = getattr(self, f'dropout{i}')
            
            x = linear(x)
            x = batchnorm(x)
            x = relu(x)
            x = dropout(x)

        # x = self.linear2(x)
        # x = self.batchnorm2(x)
        # x = self.relu2(x)
        # x = self.dropout2(x)
        
        # x = self.linear3(x)
        # x = self.batchnorm3(x)
        # x = self.relu3(x)
        # x = self.dropout3(x)
        
        # x = self.linear4(x)
        x = self.linear_out(x)

        return x

model_mlp = MLP_model(drop_prob=0.3).to(device)

#model_mlp.load_state_dict(torch.load('model_mlp_bn.pth'))

model_mlp.to('cpu')
quantized_mlp_torch = torch.quantization.quantize_dynamic(
    model_mlp, {torch.nn.Linear}, dtype=torch.qint8
)
model_mlp = model_mlp.to("cuda")

def quantize_symmetric_cuda(weight, num_bits=8):
    quantized_weight, scale = quantize_extension.quantize_symmetric_cuda(weight, num_bits)
    return quantized_weight, scale


class CustomQuantizedLinear(nn.Module):
    def __init__(self, linear_module, layer_name):
        super(CustomQuantizedLinear, self).__init__()
        self.layer_name = layer_name
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        self.bias = linear_module.bias

        # 가중치 양자화
        weight = linear_module.weight
        self.weight_int8, self.weight_scale = quantize_extension.quantize_symmetric_cuda(weight, 8)
        self.weight_int8 = self.weight_int8.to('cuda')
        self.weight_scale = self.weight_scale.to('cuda')
        self.bias = self.bias.to('cuda')
        
        # 버퍼로 등록하여 state_dict에 포함 (고유한 이름 사용)
        self.register_buffer(f'{layer_name}_weight_int8', self.weight_int8)
        self.register_buffer(f'{layer_name}_weight_scale', self.weight_scale)
        self.register_buffer(f'{layer_name}_bias', self.bias)
        
        
    def forward(self, x):
        x = x.to('cuda')
        if not x.is_cuda:
            raise ValueError("Input tensor must be on CUDA device")
    
        x_int8, x_scale = quantize_extension.quantize_symmetric_cuda(x, 8)

        # int8 행렬 곱셈을 시뮬레이션 (torch.matmul은 int8을 직접 지원하지 않음)
        weight_int8 = getattr(self, f'{self.layer_name}_weight_int8')
        
        a = x_int8
        b = weight_int8

        M = a.size(0)
        K = a.size(1)
        N = b.size(0)
        pad_multiple = 8

        # 각 차원에 대한 패딩 계산
        def calculate_padding(size, multiple):
            return (multiple - (size % multiple)) % multiple
        
        padding_M = calculate_padding(M, pad_multiple)
        padding_K = calculate_padding(K, pad_multiple)
        padding_N = calculate_padding(N, pad_multiple)

        # 패딩된 크기
        M_padded = M + padding_M
        K_padded = K + padding_K
        N_padded = N + padding_N

        # 패딩이 필요한 경우, 0으로 패딩
        if padding_M > 0:
            pad_a_M = torch.zeros((padding_M, K), dtype=torch.int8, device='cuda')
            a_padded = torch.cat([a, pad_a_M], dim=0)
        else:
            a_padded = a

        if padding_K > 0:
            pad_a_K = torch.zeros((M_padded, padding_K), dtype=torch.int8, device='cuda')
            a_padded = torch.cat([a_padded, pad_a_K], dim=1)
            
            pad_b_K = torch.zeros((N, padding_K), dtype=torch.int8, device='cuda')
            b_padded = torch.cat([b, pad_b_K], dim=1)
        else:
            b_padded = b

        if padding_N > 0:
            pad_b_N = torch.zeros((padding_N, K_padded), dtype=torch.int8, device='cuda')
            b_padded = torch.cat([b_padded, pad_b_N], dim=0)

        b_padded = b_padded.t()
        # 행렬 곱 (int32 결과)
        # print(a_padded.shape)
        # print(b_padded.shape)   
        output_int8 = quantize_extension.matmul_int8_cuda(a_padded, b_padded)
        #print(output_int8.shape)

        output_int8 = output_int8[:M, :N]
        #print(output_int8.shape)

        # 스케일 결합
        weight_scale = getattr(self, f'{self.layer_name}_weight_scale')
        output_scale = x_scale.item() * weight_scale.item()
        
        # 디양자화
        #output = output_int32.float() * output_scale
        output = output_int8.float() * output_scale
          
        bias = getattr(self, f'{self.layer_name}_bias')
        # 바이어스 추가 (부동 소수점)
        # print(output.shape)
        # print(bias.shape)
        if bias is not None:
            output += bias
        
        return output

    def __repr__(self):
        return f"CustomQuantizedLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
    

def replace_with_custom_quantized_linear(model):
    # 모델을 새로운 인스턴스로 초기화
    quantized_model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 부모 모듈 찾기
            parent = quantized_model
            names = name.split('.')
            for n in names[:-1]:
                parent = getattr(parent, n)
            linear_name = names[-1]
            # 커스텀 양자화 선형 계층으로 교체
            custom_linear = CustomQuantizedLinear(module, linear_name)
            setattr(parent, linear_name, custom_linear)
    return quantized_model

original_model = model_mlp

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

quantized_model_mlp = replace_with_custom_quantized_linear(original_model)


# 모델 크기 비교 함수
def get_size(model):
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / 1e6
    return size_mb

print(f"Original model size: {get_size(original_model):.3f} MB")
print(f"pytorch Quantized model size: {get_size(quantized_mlp_torch):.3f} MB")
print(f"custom Quantized model size: {get_size(quantized_model_mlp):.3f} MB")


# Test model and check accuracy
with torch.no_grad():
    quantized_mlp_torch.eval()    # set the model to evaluation mode (dropout=False)
    quantized_mlp_torch.cpu()
    
    # Test the model using test sets
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to('cpu')
    Y_test = mnist_test.test_labels.to('cpu')
    
    
    prediction = quantized_mlp_torch(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    #print(len(correct_prediction))
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
    
    # 임의의 자료 가져와 일치여부 확인
    r = random.randint(0, len(mnist_test) - 1)
    #print(len(mnist_test))
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float()
    Y_single_data = mnist_test.test_labels[r:r + 1]
    

    print('Label: ', Y_single_data.item())


    elapsed_time_ms = 0
    num_runs = 10
    for i in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        #start_time = time.time()
        single_prediction = quantized_mlp_torch(X_single_data)
        #end_time = time.time()
        #elapsed_time_ms = (end_time - start_time) * 1000  # 초를 밀리초로 변환
        #print("quantization using torch: {:.3f} ms".format(elapsed_time_ms))
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms += start_event.elapsed_time(end_event)
    print("quantization using torch: {:.3f} ms".format(elapsed_time_ms/num_runs))

    #print(quantized_mlp.linear1.weight())
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

# # Test model and check accuracy
# with torch.no_grad():
#     quantized_model_mlp.eval()    # set the model to evaluation mode (dropout=False)
#     quantized_model_mlp.to('cuda')

#     # Test the model using test sets
#     X_test = mnist_test.test_data.view(-1, 28 * 28).float().to('cuda').contiguous()
#     Y_test = mnist_test.test_labels.to('cuda')
    
#     prediction = quantized_model_mlp(X_test)
#     correct_prediction = torch.argmax(prediction, 1) == Y_test
#     #print(len(correct_prediction))
#     accuracy = correct_prediction.float().mean()
#     print('Accuracy:', accuracy.item())
    
#     # 임의의 자료 가져와 일치여부 확인
#     r = random.randint(0, len(mnist_test) - 1)
#     #print(len(mnist_test))
#     X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to('cuda').contiguous()
#     Y_single_data = mnist_test.test_labels[r:r + 1].to('cuda')

#     print('Label: ', Y_single_data.item())

#     elapsed_time_ms = 0
#     num_runs = 1
#     for i in range(num_runs):
#         start_event = torch.cuda.Event(enable_timing=True)
#         end_event = torch.cuda.Event(enable_timing=True)
#         start_event.record()
#         #start_time = time.time()
#         single_prediction = quantized_model_mlp(X_single_data)
#         #end_time = time.time()
#         #elapsed_time_ms = (end_time - start_time) * 1000  # 초를 밀리초로 변환
#         #print("quantization using torch: {:.3f} ms".format(elapsed_time_ms))
#         end_event.record()
#         torch.cuda.synchronize()
#         elapsed_time_ms += start_event.elapsed_time(end_event)
#     print("quantization using cuda: {:.3f} ms".format(elapsed_time_ms/num_runs))

#     print('Prediction: ', torch.argmax(single_prediction, 1).item())

def inference():
    with torch.no_grad():
        quantized_model_mlp.eval()    # set the model to evaluation mode (dropout=False)
        quantized_model_mlp.to('cuda')

        # Test the model using test sets
        X_test = mnist_test.test_data.view(-1, 28 * 28).float().to('cuda').contiguous()
        Y_test = mnist_test.test_labels.to('cuda')
        #print(X_test.shape)

        prediction = quantized_model_mlp(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        #print(len(correct_prediction))
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())

        # 임의의 자료 가져와 일치여부 확인
        r = random.randint(0, len(mnist_test) - 1)
        #print(len(mnist_test))
        X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to('cuda').contiguous()
        Y_single_data = mnist_test.test_labels[r:r + 1].to('cuda')

        print('Label: ', Y_single_data.item())

        
        elapsed_time_ms = 0
        num_runs = 10
        for i in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            #start_time = time.time()
            single_prediction = quantized_model_mlp(X_single_data)
            #end_time = time.time()
            #elapsed_time_ms = (end_time - start_time) * 1000  # 초를 밀리초로 변환
            #print("quantization using torch: {:.3f} ms".format(elapsed_time_ms))
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)
        print("quantization using cuda: {:.3f} ms".format(elapsed_time_ms/num_runs))

        print('Prediction: ', torch.argmax(single_prediction, 1).item())


if __name__ == "__main__":
    inference()