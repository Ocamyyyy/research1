import socket
import argparse
import time
import sys


class UdpConnectionTester:
    def __init__(self, target_host, target_port, timeout=3, retries=3):
        self.target_host = target_host
        self.target_port = target_port
        self.timeout = timeout
        self.retries = retries
        self.sock = None

    def create_socket(self):
        """创建UDP socket"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(self.timeout)
            return True
        except Exception as e:
            print(f"创建Socket失败: {e}")
            return False

    def test_connection(self, test_message="TEST_CONNECTION"):
        """测试UDP连接"""
        print(f"开始测试UDP连接: {self.target_host}:{self.target_port}")
        print(f"超时时间: {self.timeout}秒, 重试次数: {self.retries}")
        print("-" * 50)

        if not self.create_socket():
            return False

        test_data = test_message.encode('utf-8')
        response_received = False
        response_time = None

        for attempt in range(1, self.retries + 1):
            try:
                print(f"尝试 {attempt}/{self.retries}: 发送测试包...", end=" ")
                sys.stdout.flush()

                # 记录发送时间
                start_time = time.time()
                self.sock.sendto(test_data, (self.target_host, self.target_port))

                # 尝试接收响应
                data, addr = self.sock.recvfrom(1024)
                end_time = time.time()

                response_time = round((end_time - start_time) * 1000, 2)  # 毫秒
                response_received = True

                # 解析响应
                try:
                    response_text = data.decode('utf-8').strip()
                    print(f"成功! 收到来自 {addr[0]}:{addr[1]} 的响应")
                    print(f"响应时间: {response_time} ms")
                    print(f"响应内容: '{response_text}'")
                except UnicodeDecodeError:
                    hex_data = ' '.join(f'{b:02x}' for b in data[:32])
                    if len(data) > 32:
                        hex_data += '...'
                    print(f"成功! 收到来自 {addr[0]}:{addr[1]} 的响应")
                    print(f"响应时间: {response_time} ms")
                    print(f"响应数据(十六进制): {hex_data} ({len(data)} 字节)")

                break

            except socket.timeout:
                print("超时 - 未收到响应")
                continue
            except ConnectionRefusedError:
                print("连接被拒绝 - 目标可能未监听该端口")
                break
            except Exception as e:
                print(f"错误: {e}")
                break

        return response_received, response_time

    def advanced_test(self, port_range=None, multiple_messages=False):
        """高级测试功能"""
        print("\n" + "=" * 60)
        print("高级UDP连接测试")
        print("=" * 60)

        # 测试1: 基本连接测试
        print("\n1. 基本连接测试:")
        success, rtt = self.test_connection()

        if not success:
            print("\n❌ 基本连接测试失败 - 目标可能未响应")
            return False

        print("\n✅ 基本连接测试成功!")

        # 测试2: 多消息测试（可选）
        if multiple_messages:
            print("\n2. 多消息响应测试:")
            test_messages = ["PING", "HELLO", "TEST123"]
            for msg in test_messages:
                success, rtt = self.test_connection(msg)
                if success:
                    print(f"  消息 '{msg}' 测试成功")
                else:
                    print(f"  消息 '{msg}' 测试失败")
                time.sleep(0.5)

        # 测试3: 端口扫描（可选）
        if port_range:
            print(f"\n3. 端口扫描测试 ({port_range[0]}-{port_range[1]}):")
            self.port_scan(port_range)

        return True

    def port_scan(self, port_range):
        """简单的端口扫描"""
        start_port, end_port = port_range
        print(f"扫描端口 {start_port}-{end_port}...")

        open_ports = []
        for port in range(start_port, end_port + 1):
            if port == self.target_port:
                continue  # 跳过已经测试的主端口

            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                test_sock.settimeout(1)
                test_sock.sendto(b"PORT_TEST", (self.target_host, port))
                data, addr = test_sock.recvfrom(1024)
                open_ports.append(port)
                print(f"  端口 {port}: 开放 ✅")
                test_sock.close()
            except (socket.timeout, ConnectionRefusedError):
                pass  # 端口关闭或超时
            except Exception:
                pass
            finally:
                if 'test_sock' in locals():
                    test_sock.close()

        if open_ports:
            print(f"发现开放端口: {open_ports}")
        else:
            print("未发现其他开放UDP端口")

    def close(self):
        """关闭socket"""
        if self.sock:
            self.sock.close()


def main():
    parser = argparse.ArgumentParser(description='UDP连接测试工具')
    parser.add_argument('host', help='目标主机IP地址')
    parser.add_argument('port', type=int, help='目标端口号')
    parser.add_argument('-t', '--timeout', type=float, default=3,
                        help='超时时间(秒)，默认3秒')
    parser.add_argument('-r', '--retries', type=int, default=3,
                        help='重试次数，默认3次')
    parser.add_argument('-a', '--advanced', action='store_true',
                        help='执行高级测试（多消息测试）')
    parser.add_argument('-p', '--port-scan', type=int, nargs=2,
                        metavar=('START', 'END'),
                        help='扫描端口范围，例如: -p 8000 9000')

    args = parser.parse_args()

    # 创建测试器实例
    tester = UdpConnectionTester(args.host, args.port, args.timeout, args.retries)

    try:
        if args.advanced or args.port_scan:
            # 高级测试模式
            tester.advanced_test(args.port_scan, args.advanced)
        else:
            # 基本测试模式
            success, rtt = tester.test_connection()

            print("\n" + "=" * 50)
            if success:
                print(f"✅ UDP连接测试成功!")
                print(f"目标: {args.host}:{args.port}")
                print(f"响应时间: {rtt} ms")
                print("状态: 连接正常 ✅")
            else:
                print(f"❌ UDP连接测试失败!")
                print(f"目标: {args.host}:{args.port}")
                print("可能的原因:")
                print("  - 目标主机未运行")
                print("  - 目标端口未监听")
                print("  - 防火墙阻止了通信")
                print("  - 网络连接问题")
                print("状态: 连接失败 ❌")

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试错误: {e}")
    finally:
        tester.close()


if __name__ == "__main__":
    main()