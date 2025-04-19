# test_env.py
import torch

def main():
    print("PyTorch version:", torch.__version__)

    cuda_available = torch.cuda.is_available()
    print("CUDA kullanılabilir mi? ", cuda_available)

    if cuda_available:
        print("CUDA cihaz sayısı: ", torch.cuda.device_count())
        print("Geçerli cihaz: ", torch.cuda.current_device(), "-", torch.cuda.get_device_name(0))

        # Basit bir tensor işlemiyle GPU’yu test edelim
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        z = x @ y   # matris çarpımı
        print("GPU üzerinde çarpım sonucu tensor boyutu:", z.size())

        # Sonuç CPU’ya alıp küçük bir kontrol
        z_cpu = z.to("cpu")
        print("Sonuç CPU’ya alındı, örnek değer [0,0]:", z_cpu[0,0].item())
    else:
        print("CUDA bulunamadı, ortam GPU destekli değil veya sürücüler eksik.")

if __name__ == "__main__":
    main()
