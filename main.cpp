#define _CRT_SECURE_NO_WARNINGS

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>

#include <Eigen/Dense>
#include <MiniDNN.h>

class CustomCallback : public MiniDNN::Callback
{
public:
    void post_training_batch( const MiniDNN::Network* net, const Matrix& x, const Matrix& y )
    {
        if ( m_batch_id < m_nbatch - 1 ) return;
        const MiniDNN::Scalar loss = net->get_output()->loss();
        std::cout << "[Epoch " << m_epoch_id << "] Loss = " << loss << std::endl;
    }

    void post_training_batch( const MiniDNN::Network* net, const Matrix& x, const IntegerVector& y )
    {
        if ( m_batch_id < m_nbatch - 1 ) return;
        MiniDNN::Scalar loss = net->get_output()->loss();
        std::cout << "[Epoch " << m_epoch_id << "] Loss = " << loss << std::endl;
    }
};

Eigen::MatrixXd load_csv( const std::string& path, int columns )
{
    FILE* file = std::fopen( path.data(), "r" );
    assert( file != nullptr );

    std::vector<double> values;
    while ( !std::feof( file ) )
    {
        char _;
        for ( int i = 0; i < columns; ++i )
        {
            double value = 0.0;
            std::fscanf( file, "%le", &value );
            if ( i < columns - 1 ) std::fscanf( file, "%c", &_ );
            values.push_back( value );
        }
    }

    std::fclose( file );
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        values.data(), values.size() / columns, columns );
}

Eigen::MatrixXd pick_data( Eigen::MatrixXd& matrix, double* max, double* min )
{
    int ncomps = matrix.rows() / 2 - 1;

    std::vector<int> index;
    for ( int i = 0; i < matrix.cols(); ++i )
    {
        // if ( ( matrix.col( i ).topRows( ncomps ).array() < 1.0e-06 ).any() ) continue;
        if ( matrix( ncomps + 1, i ) < 1.0e-06 || matrix( ncomps + 1, i ) > ( 1.0 - 1.0e-06 ) )
            continue;
        if ( ( matrix.col( i ).bottomRows( ncomps ).array() < 1.0e-30 ).any() ) continue;
        if ( ( abs( log( matrix.col( i ).bottomRows( ncomps ).array() ) ) < 1.0e-05 ).any() )
            continue;
        index.push_back( i );
    }

    Eigen::MatrixXd temp = matrix( Eigen::all, index );
    for ( int i = 0; i < temp.rows(); ++i )
    {
        if ( temp.row( i ).maxCoeff() > max[i] ) max[i] = temp.row( i ).maxCoeff();
        if ( temp.row( i ).minCoeff() < min[i] ) min[i] = temp.row( i ).minCoeff();
    }
    return temp;
}

Eigen::MatrixXd transform_data( Eigen::MatrixXd& matrix, double* max, double* min )
{
    int ncomps = matrix.rows() / 2 - 1;
    for ( int i = 0; i < ncomps + 2; ++i )
    {
        // double max = matrix.row( i ).maxCoeff();
        // double min = matrix.row( i ).minCoeff();

        matrix.array().row( i ) -= min[i];
        matrix.array().row( i ) /= max[i] - min[i];
    }

    for ( int i = ncomps + 2; i < matrix.rows(); ++i )
    {
        // double max = matrix.row( i ).maxCoeff();
        // double min = matrix.row( i ).minCoeff();

        matrix.array().row( i ) = log( matrix.array().row( i ) );
        matrix.array().row( i ) -= log( min[i] );
        matrix.array().row( i ) /= log( max[i] ) - log( min[i] );
        matrix.array().row( i ) = sqrt( matrix.array().row( i ) );
    }

    return matrix;
}

int main( int argc, char* argv[] )
{
    std::srand( 123 );
    if ( argc > 2 )
    {
        const int ncomps = std::stoi( argv[1] );
        assert( ncomps > 1 );
        double* max = new double[2 * ncomps + 2];
        double* min = new double[2 * ncomps + 2];
        std::fill( max, max + 2 * ncomps + 2, std::numeric_limits<double>::epsilon() );
        std::fill( min, min + 2 * ncomps + 2, std::numeric_limits<double>::infinity() );

        std::filesystem::path file( argv[2] );
        std::filesystem::path dir = file.parent_path();
        if ( !dir.empty() )
        {
            std::cout << "Switch working directory to \"" << dir << "\"" << std::endl;
            std::filesystem::current_path( file.parent_path() );
        }

        Eigen::MatrixXd raw   = load_csv( argv[2], 2 * ( ncomps + 1 ) ).transpose();
        Eigen::MatrixXd train = pick_data( raw, max, min );


        raw                  = load_csv( argv[3], 2 * ( ncomps + 1 ) ).transpose();
        Eigen::MatrixXd test = pick_data( raw, max, min );


        raw                 = load_csv( argv[4], 2 * ( ncomps + 1 ) ).transpose();
        Eigen::MatrixXd val = pick_data( raw, max, min );

        transform_data( train, max, min );
        transform_data( test, max, min );
        transform_data( val, max, min );
        // std::cout << train.col( 0 ) << std::endl;
        MiniDNN::Layer*  layer1 = new MiniDNN::FullyConnected<MiniDNN::Softmax>( ncomps + 1, 10 );
        MiniDNN::Layer*  layer2 = new MiniDNN::FullyConnected<MiniDNN::Softmax>( 10, 10 );
        MiniDNN::Layer*  layer3 = new MiniDNN::FullyConnected<MiniDNN::Softmax>( 10, ncomps + 1 );
        MiniDNN::Output* output = new MiniDNN::RegressionMSE();
        MiniDNN::Network net;
        net.add_layer( layer1 );
        net.add_layer( layer2 );
        net.add_layer( layer3 );
        net.set_output( output );
        net.init( 0, 0.01, 123 );
        CustomCallback callback;
        net.set_callback( callback );
        MiniDNN::Adam opt( 0.1, 1e-7 );

        net.fit( opt, train.topRows( ncomps + 1 ), train.bottomRows( ncomps + 1 ), 3000, 500, 123 );
        // train.bottomRows( ncomps + 1 ) = net.predict( train.topRows( ncomps + 1 ) );
        std::cout << val.col( 0 ).topRows( ncomps + 1 ) << std::endl;
        std::cout << val.col( 0 ).bottomRows( ncomps + 1 ) << std::endl;
        std::cout << net.predict( val.col( 0 ).topRows( ncomps + 1 ) ) << std::endl;
        std::cout << net.predict( val.col( 15 ).topRows( ncomps + 1 ) ) << std::endl;
    }
    else
    {
        std::cerr << "Invalid input data was specified" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
