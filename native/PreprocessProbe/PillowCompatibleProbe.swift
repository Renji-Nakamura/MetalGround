import Foundation
import Metal

struct RefMetadata: Decodable {
    struct Input: Decodable { let width:Int; let height:Int; let raw_file:String }
    struct Letterbox: Decodable { let width:Int; let height:Int; let fill_u8:Int; let resized_width:Int; let resized_height:Int; let left:Int; let top:Int }
    struct Processor: Decodable { let output_shape_chw:[Int]; let image_mean:[Float]; let image_std:[Float] }
    let input:Input; let letterbox:Letterbox; let processor:Processor
}
struct CMeta: Decodable { struct S:Decodable{let in_size:Int;let out_size:Int;let ksize:Int;let bounds_file:String;let coeffs_file:String;let precision_bits:Int}; let precision_bits:Int; let stages:[String:S] }
struct R { var sw:UInt32; var sh:UInt32; var dw:UInt32; var dh:UInt32; var k:UInt32 }
struct LV { var sw:UInt32;var sh:UInt32;var dw:UInt32;var dh:UInt32;var rh:UInt32;var oy:UInt32;var k:UInt32;var fill:UInt32 }
struct FV { var sw:UInt32;var sh:UInt32;var dw:UInt32;var dh:UInt32;var k:UInt32 }
func pct(_ x:[Double],_ p:Double)->Double{let a=x.sorted();let q=p*Double(a.count-1),l=Int(floor(q)),h=Int(ceil(q));if l==h{return a[l]};let t=q-Double(l);return a[l]*(1-t)+a[h]*t}
func stats(_ x:[Double])->[String:Any]{["n":x.count,"mean":x.reduce(0,+)/Double(x.count),"median":pct(x,0.5),"p90":pct(x,0.9),"p95":pct(x,0.95),"min":x.min()!,"max":x.max()!]}
func dispatch(_ e:MTLComputeCommandEncoder,_ p:MTLComputePipelineState,_ w:Int,_ h:Int){let tw=min(p.threadExecutionWidth,w),th=max(1,min(p.maxTotalThreadsPerThreadgroup/tw,h));e.dispatchThreads(MTLSize(width:w,height:h,depth:1),threadsPerThreadgroup:MTLSize(width:tw,height:th,depth:1))}
func buf(_ d:MTLDevice,_ data:Data)->MTLBuffer{let b=d.makeBuffer(length:data.count,options:[.storageModeShared])!;data.copyBytes(to:b.contents().assumingMemoryBound(to:UInt8.self),count:data.count);return b}
@main struct Main {
 static func main() throws {
  let a=CommandLine.arguments;func arg(_ n:String,_ f:String)->String{if let i=a.firstIndex(of:n),i+1<a.count{return a[i+1]};return f}
  let rd=URL(fileURLWithPath:arg("--reference-dir","results/0054b_reference"));let cd=URL(fileURLWithPath:arg("--coeff-dir","results/0054b_pillow_coeffs"));let ms=arg("--metal-source","native/PreprocessProbe/PillowCompatibleResize.metal");let out=arg("--output","results/metalground_metal_preprocess_0054b2.json");let warm=Int(arg("--warmup","20"))!,n=Int(arg("--samples","200"))!
  let ref=try JSONDecoder().decode(RefMetadata.self,from:Data(contentsOf:rd.appendingPathComponent("metadata.json")));let cm=try JSONDecoder().decode(CMeta.self,from:Data(contentsOf:cd.appendingPathComponent("metadata.json")));guard cm.precision_bits==22 else{fatalError("precision")}
  let s1h=cm.stages["stage1_h"]!,s1v=cm.stages["stage1_v"]!,s2h=cm.stages["stage2_h"]!,s2v=cm.stages["stage2_v"]!
  let d=MTLCreateSystemDefaultDevice()!,q=d.makeCommandQueue()!,src=try String(contentsOf:URL(fileURLWithPath:ms),encoding:.utf8),lib=try d.makeLibrary(source:src,options:MTLCompileOptions())
  let p1h=try d.makeComputePipelineState(function:lib.makeFunction(name:"h1")!),p1v=try d.makeComputePipelineState(function:lib.makeFunction(name:"v1_letter")!),p2h=try d.makeComputePipelineState(function:lib.makeFunction(name:"h2")!),p2v=try d.makeComputePipelineState(function:lib.makeFunction(name:"v2_norm")!)
  let input=buf(d,try Data(contentsOf:rd.appendingPathComponent(ref.input.raw_file)))
  func load(_ s:CMeta.S)->(MTLBuffer,MTLBuffer){(buf(d,try! Data(contentsOf:cd.appendingPathComponent(s.bounds_file))),buf(d,try! Data(contentsOf:cd.appendingPathComponent(s.coeffs_file))))}
  let (b1h,c1h)=load(s1h),(b1v,c1v)=load(s1v),(b2h,c2h)=load(s2h),(b2v,c2v)=load(s2v)
  let t1=d.makeBuffer(length:1200*720*4,options:[.storageModeShared])!,letter=d.makeBuffer(length:1200*901*4,options:[.storageModeShared])!,t2=d.makeBuffer(length:1065*901*4,options:[.storageModeShared])!,oc=3*800*1065,outb=d.makeBuffer(length:oc*4,options:[.storageModeShared])!
  var r1=R(sw:1280,sh:720,dw:1200,dh:720,k:UInt32(s1h.ksize)),lv=LV(sw:1200,sh:720,dw:1200,dh:901,rh:675,oy:113,k:UInt32(s1v.ksize),fill:UInt32(ref.letterbox.fill_u8)),r2=R(sw:1200,sh:901,dw:1065,dh:901,k:UInt32(s2h.ksize)),fv=FV(sw:1065,sh:901,dw:1065,dh:800,k:UInt32(s2v.ksize));var mean=SIMD3<Float>(ref.processor.image_mean[0],ref.processor.image_mean[1],ref.processor.image_mean[2]),std=SIMD3<Float>(ref.processor.image_std[0],ref.processor.image_std[1],ref.processor.image_std[2])
  func run()->Double{let t0=DispatchTime.now().uptimeNanoseconds;let cb=q.makeCommandBuffer()!;let e1=cb.makeComputeCommandEncoder()!;e1.setComputePipelineState(p1h);e1.setBuffer(input,offset:0,index:0);e1.setBuffer(t1,offset:0,index:1);e1.setBuffer(b1h,offset:0,index:2);e1.setBuffer(c1h,offset:0,index:3);e1.setBytes(&r1,length:MemoryLayout<R>.stride,index:4);dispatch(e1,p1h,1200,720);e1.endEncoding();let e2=cb.makeComputeCommandEncoder()!;e2.setComputePipelineState(p1v);e2.setBuffer(t1,offset:0,index:0);e2.setBuffer(letter,offset:0,index:1);e2.setBuffer(b1v,offset:0,index:2);e2.setBuffer(c1v,offset:0,index:3);e2.setBytes(&lv,length:MemoryLayout<LV>.stride,index:4);dispatch(e2,p1v,1200,901);e2.endEncoding();let e3=cb.makeComputeCommandEncoder()!;e3.setComputePipelineState(p2h);e3.setBuffer(letter,offset:0,index:0);e3.setBuffer(t2,offset:0,index:1);e3.setBuffer(b2h,offset:0,index:2);e3.setBuffer(c2h,offset:0,index:3);e3.setBytes(&r2,length:MemoryLayout<R>.stride,index:4);dispatch(e3,p2h,1065,901);e3.endEncoding();let e4=cb.makeComputeCommandEncoder()!;e4.setComputePipelineState(p2v);e4.setBuffer(t2,offset:0,index:0);e4.setBuffer(outb,offset:0,index:1);e4.setBuffer(b2v,offset:0,index:2);e4.setBuffer(c2v,offset:0,index:3);e4.setBytes(&fv,length:MemoryLayout<FV>.stride,index:4);e4.setBytes(&mean,length:MemoryLayout<SIMD3<Float>>.stride,index:5);e4.setBytes(&std,length:MemoryLayout<SIMD3<Float>>.stride,index:6);dispatch(e4,p2v,1065,800);e4.endEncoding();cb.commit();cb.waitUntilCompleted();if cb.status != .completed{fatalError("Metal \(String(describing:cb.error))")};return Double(DispatchTime.now().uptimeNanoseconds-t0)/1e6}
  for _ in 0..<warm{_ = run()};var ts:[Double]=[];for _ in 0..<n{ts.append(run())}
  let ou=URL(fileURLWithPath:out),od=ou.deletingLastPathComponent();try FileManager.default.createDirectory(at:od,withIntermediateDirectories:true);let tp=od.appendingPathComponent("metal_pixel_values_f32_chw_0054b2.bin"),lp=od.appendingPathComponent("metal_letterbox_rgba8_0054b2.raw");try Data(bytes:outb.contents(),count:oc*4).write(to:tp);try Data(bytes:letter.contents(),count:1200*901*4).write(to:lp)
  let res:[String:Any]=["experiment":"0054b-2","metal_device":d.name,"timing_ms":stats(ts),"output_file":tp.path,"diagnostic_letterbox_file":lp.path,"output_shape_chw":[3,800,1065],"implementation":["precision_bits":22,"hardware_linear_sampler_used":false,"approximation":false,"reduced_precision_output":false],"timing_semantics":"submit-to-completion for four kernels; setup excluded"]
  try JSONSerialization.data(withJSONObject:res,options:[.prettyPrinted,.sortedKeys]).write(to:ou);let st=res["timing_ms"] as! [String:Any];print("=== 0054b-2 Pillow-compatible Metal preprocess ===");print(String(format:"median %.4f ms, p95 %.4f ms",st["median"] as! Double,st["p95"] as! Double));print("Saved:",out)
 }
}
