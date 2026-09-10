module adder(
    input  wire [3:0] a,
    input  wire [3:0] b,
    output wire [4:0] sum
);
    assign sum = a + b;
endmodule

module counter(
    input wire clk,
    input wire reset,
    output reg [7:0] count
);
    // increments on every clock edge, resets to zero
    always @(posedge clk or posedge reset) begin
        if (reset)
            count <= 8'b0;
        else
            count <= count + 1;
    end
endmodule
